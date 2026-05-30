import os
import sys
import numpy as np
import argparse
from PIL import Image
import cv2
from tqdm import tqdm, trange
import logging

try:
    from cocoplus.coco import COCO_PLUS
except ImportError:
    print("ERROR: Failed to import COCO_PLUS. Make sure it's installed or accessible in your Python path.")
    sys.exit(1)
try:
    # from rfdetr import RFDETRBase
    from rfdetr import RFDETRLarge
    from rfdetr.util.coco_classes import COCO_CLASSES
except ImportError:
    print("ERROR: Failed to import RFDETRBase or COCO_CLASSES from rfdetr. Make sure rfdetr is installed.")
    sys.exit(1)
from pynuscenes.nuscenes_dataset import NuscenesDataset
from nuscenes.utils.geometry_utils import view_points

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Converts NuScenes dataset to COCO format using RF-DETR predictions.')
    parser.add_argument('--nusc_root', default='../data/nuscenes', help='NuScenes dataroot path.')
    parser.add_argument('--split', default='mini_train', help='NuScenes split (mini_train, mini_val, train, val).')
    parser.add_argument('--out_dir', default='../data/nucoco/', help='Output base directory for COCO dataset.')
    parser.add_argument('--nsweeps_radar', default=1, type=int, help='Number of Radar sweeps (if processing point clouds).')
    parser.add_argument('--cameras', nargs='+', default=['CAM_FRONT', 'CAM_BACK'], help='Cameras to include.')
    parser.add_argument('--confidence_threshold', type=float, default=0.3, help='Confidence threshold for RF-DETR predictions.')
    parser.add_argument('-l', '--logging_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging level.')
    args = parser.parse_args()
    log_level_map = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR, 'CRITICAL': logging.CRITICAL}
    logging.getLogger().setLevel(log_level_map.get(args.logging_level.upper(), logging.INFO))
    return args

def main():
    args = parse_args()
    logger.info(f"Starting NuScenes to COCO conversion using RF-DETR for split: {args.split}")
    logger.info("Loading RF-DETR model...")
    try:
        # rfdetr_model = RFDETRBase()
        # logger.info("RF-DETR model loaded.")
        rfdetr_model = RFDETRLarge()
        logger.info(f"RF-DETR model type: {type(rfdetr_model)}")
    except Exception as e:
        logger.critical(f"Failed load RFDETR: {e}", exc_info=True)
        sys.exit(1)

    if not isinstance(COCO_CLASSES, dict): logger.critical(f"COCO_CLASSES not dict."); sys.exit(1)
    logger.info(f"Using COCO_CLASSES dict from rfdetr with {len(COCO_CLASSES)} entries.")
    target_obstacle_names = ['bicycle', 'car', 'motorcycle', 'bus', 'person', 'truck']
    target_indices_set=set(); model_id_to_name={}; category_map={}; new_categories=[]
    output_cat_id_counter = 0
    for model_id, name in COCO_CLASSES.items():
        try: model_id_int = int(model_id); model_id_to_name[model_id_int] = name
        except ValueError: continue
        if name in target_obstacle_names: target_indices_set.add(model_id_int)
    for name in target_obstacle_names:
        found_id = -1
        for model_id, current_name in COCO_CLASSES.items():
             if current_name == name:
                 try: found_id = int(model_id); break
                 except ValueError: continue
        if found_id != -1:
            category_map[found_id] = output_cat_id_counter; supercategory = 'vehicle'
            new_categories.append({"id": output_cat_id_counter, "name": name, "supercategory": supercategory})
            output_cat_id_counter += 1
        else: logger.warning(f"Target name '{name}' not found in COCO_CLASSES dict.")
    logger.info(f"RF-DETR Target Indices Set (for filtering): {target_indices_set}")
    logger.info(f"Output JSON Categories: {new_categories}")
    logger.info(f"RF-DETR Model ID -> Output JSON Category ID Map: {category_map}")
    if not new_categories: logger.error("No target categories mapped."); sys.exit(1)

    if "mini" in args.split: nusc_version = "v1.0-mini"
    elif "val" in args.split: nusc_version = "v1.0-trainval"
    elif "train" in args.split: nusc_version = "v1.0-trainval"
    else: logger.error(f"Unrecognized split: {args.split}"); sys.exit(1)

    logger.info(f"Loading NuScenes {nusc_version} dataset for split '{args.split}'...")
    try:
        sensors_to_return = ['camera']; process_radar = args.nsweeps_radar > 0
        if process_radar: sensors_to_return.append('radar')
        nusc_dataset = NuscenesDataset(nusc_path=args.nusc_root, nusc_version=nusc_version, split=args.split,
                                       coordinates='vehicle', nsweeps_radar=args.nsweeps_radar,
                                       sensors_to_return=sensors_to_return, pc_mode='camera',
                                       logging_level=args.logging_level.upper())
        nusc = nusc_dataset.nusc
        logger.info(f"NuscenesDataset loaded with {len(nusc_dataset)} samples.")
    except Exception as e: logger.critical(f"Failed load NuscenesDataset: {e}", exc_info=True); sys.exit(1)

    coco_dataset = COCO_PLUS(logging_level=args.logging_level.upper())
    coco_dataset.create_new_dataset(dataset_dir=args.out_dir, split=args.split)
    for cat_info in new_categories:
        coco_dataset.addCategory(cat_info['name'], cat_info.get('supercategory', 'object'), cat_info['id'])

    num_samples = len(nusc_dataset)
    logger.info(f"Processing {num_samples} samples from NuScenes dataset...")
    coco_image_id_counter = 0

    for i in trange(num_samples, desc=f"Processing {args.split}"):
        try:
            sample = nusc_dataset[i]
            for j, cam_sample in enumerate(sample.get('camera', [])):
                cam_name = cam_sample.get('camera_name')
                if not cam_name or cam_name not in args.cameras: continue

                cam_cs_record = cam_sample.get('cs_record')
                if not cam_cs_record or not isinstance(cam_cs_record, dict): logger.warning(f"No cs_record {i}. Skip."); continue
                sd_token = cam_cs_record.get('token')
                cam_intrinsic = cam_cs_record.get('camera_intrinsic')
                relative_img_path = cam_sample.get('cam_path') # Use cam_path
                timestamp = cam_sample.get('timestamp') or cam_cs_record.get('timestamp')
                if not sd_token or not cam_intrinsic or not relative_img_path: logger.warning(f"Missing token/intrinsic/cam_path {i}. Skip."); continue
                img_path = os.path.join(nusc.dataroot, relative_img_path)
                if not os.path.exists(img_path): logger.warning(f"No image file: {img_path}. Skip."); continue
                try: image_pil = Image.open(img_path).convert("RGB"); img_width, img_height = image_pil.size
                except Exception as e: logger.error(f"Failed load image {img_path}: {e}"); continue

                detections = None
                try: detections = rfdetr_model.predict(image_pil, threshold=args.confidence_threshold)
                except Exception as e: logger.error(f"RF-DETR predict failed {relative_img_path}:{e}"); continue

                sample_anns = []
                if detections and hasattr(detections, 'class_id') and detections.class_id is not None:
                    num_detections = len(detections.class_id)
                    if num_detections > 0 and hasattr(detections, 'xyxy') and hasattr(detections, 'confidence'):
                         try:
                            for det_idx in range(num_detections):
                                model_output_class_id = int(detections.class_id[det_idx])
                                if model_output_class_id in target_indices_set:
                                    if model_output_class_id not in category_map: continue
                                    json_category_id = category_map[model_output_class_id]
                                    score = detections.confidence[det_idx]; box_xyxy = detections.xyxy[det_idx]
                                    xmin, ymin, xmax, ymax = map(float, box_xyxy)
                                    xmin=max(0.0,xmin); ymin=max(0.0,ymin); xmax=min(float(img_width),xmax); ymax=min(float(img_height),ymax)
                                    w = xmax - xmin; h = ymax - ymin
                                    if w <= 0 or h <= 0: continue
                                    coco_bbox = [xmin, ymin, w, h]
                                    coco_ann = coco_dataset.createAnn(bbox=coco_bbox, cat_id=json_category_id, img_id=coco_image_id_counter)
                                    coco_ann['score'] = float(score)
                                    sample_anns.append(coco_ann)
                         except Exception as e: logger.error(f"Err processing dets {relative_img_path}: {e}", exc_info=True)
                    elif num_detections > 0: logger.warning(f"Dets missing xyxy/conf for {relative_img_path}")

                if not sample_anns: logger.debug(f"No target anns for {relative_img_path}."); continue

                pc_coco = []
                if process_radar and 'radar' in sample and j < len(sample.get('radar',[])):
                    try:
                        pc = sample['radar'][j]; transform_info = {'camera_intrinsic': cam_intrinsic}
                        if cam_cs_record: transform_info.update(cam_cs_record)
                        pc_cam = nusc_dataset.pc_to_sensor(pc, transform_info)
                        if pc_cam.shape[1] > 0:
                           pc_depth=pc_cam[2,:]; pc_image=view_points(pc_cam[:3,:],np.array(cam_intrinsic),normalize=True)
                           valid=(pc_image[0,:]>=0)&(pc_image[0,:]<img_width)&(pc_image[1,:]>=0)&(pc_image[1,:]<img_height)&(pc_depth>0)
                           if np.any(valid): pc_image=pc_image[:,valid]; pc_depth=pc_depth[valid]; pc_coco=np.vstack((pc_image[:2,:],pc_depth)).T.tolist()
                    except Exception as e: logger.warning(f"Error processing point cloud for {relative_img_path}: {e}")

                try:
                    image_np_bgr = np.array(image_pil)[:, :, ::-1].copy()
                    other_data = {}
                    if cam_cs_record: other_data.update(cam_cs_record)
                    other_data['token'] = sd_token
                    other_data['filename'] = relative_img_path
                    other_data['camera_intrinsic'] = cam_intrinsic
                    if timestamp: other_data['timestamp'] = timestamp

                    coco_dataset.addSample(
                        img=image_np_bgr,
                        anns=sample_anns,
                        pointcloud=pc_coco,
                        img_id=coco_image_id_counter,
                        other=other_data,
                        img_format='BGR',
                        write_img=True
                    )
                    coco_image_id_counter += 1

                except Exception as e:
                    logger.error(f"Failed add sample token {sd_token} to COCO object: {e}", exc_info=True)

        except KeyboardInterrupt: logger.warning("KeyboardInterrupt."); break
        except Exception as e: logger.error(f"Critical error sample index {i}: {e}", exc_info=True); continue

    logger.info("Finished processing samples. Saving final annotation file...")
    final_img_count = len(coco_dataset.dataset.get('images', []))
    final_ann_count = len(coco_dataset.dataset.get('annotations', []))
    logger.info(f"Saving annotations for split: {args.split} ({final_img_count} images, {final_ann_count} annotations)...")
    try:
        coco_dataset.saveAnnsToDisk()
        logger.info(f"Successfully saved {coco_dataset.ann_path}")
    except Exception as e: logger.error(f"Failed save final annotations split {args.split}: {e}", exc_info=True)

    logger.info("COCO conversion using RF-DETR complete.")

if __name__ == '__main__':
    main()