# File: nuscenes_to_coco_split.py (Corrected Version - Revision 2)

# import _init_path # noqa: F401 Ensures project modules are found
import os
import sys
# import pickle # No longer needed in this version
import numpy as np
import argparse
import cv2
from tqdm import tqdm, trange
# --- Ensure COCO_PLUS is importable from your project structure ---
# If cocoplus is a directory at the same level as your script:
# from cocoplus.coco import COCO_PLUS
# Or adjust the path/import as necessary based on your project layout
try:
    # Assuming cocoplus is installable or in the python path
    from cocoplus.coco import COCO_PLUS
except ImportError:
    # If it's a local directory structure, adjust as needed
    # e.g., from ..cocoplus.coco import COCO_PLUS
    print("ERROR: Failed to import COCO_PLUS. Make sure it's installed or accessible.")
    sys.exit(1)
# --------------------------------------------------------------------
from pynuscenes.utils.nuscenes_utils import nuscenes_box_to_coco, nuscene_cat_to_coco
from pynuscenes.nuscenes_dataset import NuscenesDataset
from nuscenes.utils.geometry_utils import view_points

# Setup basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Define the key name used to store the token in the COCO JSON ---
NUSCENES_TOKEN_KEY = "nusc_token"
# ---

def parse_args():
    parser = argparse.ArgumentParser(description='Converts the NuScenes dataset to COCO format, splitting into night/rain/other and storing NuScenes token.')

    parser.add_argument('--nusc_root', required=True,
                        help='NuScenes dataroot path (e.g., /path/to/nuscenes).')

    parser.add_argument('--split', required=True, choices=['val', 'train', 'mini_train', 'mini_val'],
                        help='Which NuScenes dataset split to process.')

    parser.add_argument('--out_dir', required=True,
                        help='Output base directory for the nucoco dataset splits (e.g., ../data/nucoco/).')

    parser.add_argument('--nsweeps_radar', default=1, type=int,
                        help='Number of Radar sweeps to include.')

    parser.add_argument('--cameras', nargs='+',
                        default=['CAM_FRONT', 'CAM_BACK'],
                        help='List of cameras to include.')

    parser.add_argument('-l', '--logging_level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level.')

    args = parser.parse_args()
    logging.getLogger().setLevel(args.logging_level)
    return args

# Helper function to process and add a single sample to a COCO object
# (Keep the process_and_add_sample function exactly as in the previous corrected version
#  - it correctly uses addSample and does not call getTotalImages)
def process_and_add_sample(coco_obj, sample, nusc_dataset, cameras_to_use, categories):
    """
    Processes a single NuScenes sample and adds relevant camera views,
    annotations, and the NuScenes sample_data_token to the provided COCO_PLUS object.
    (Identical to previous version)
    """
    nusc = nusc_dataset.nusc
    img_ids_from_sample = sample.get('img_id', []) # Sequential IDs assigned by pynuscenes

    if not img_ids_from_sample:
         logger.warning(f"Sample {sample.get('sample_idx', 'N/A')} missing 'img_id'. Skipping.")
         return 0 # Return count of images added (0)

    images_added_this_sample = 0
    printed_debug_keys = True # Keep the debug flag logic for now if needed

    for j, cam_sample in enumerate(sample.get('camera', [])):
        cam_name = cam_sample.get('camera_name')
        if not cam_name or cam_name not in cameras_to_use:
            continue

        # --- DEBUG Block (Keep or remove as desired) ---
        if not printed_debug_keys:
            logger.info(f"DEBUG: Keys available in cam_sample for cam '{cam_name}', sample index {sample.get('sample_idx', 'N/A')}: {list(cam_sample.keys())}")
            # Check keys inside cs_record too, just once
            if 'cs_record' in cam_sample and isinstance(cam_sample['cs_record'], dict):
                 logger.info(f"DEBUG: Keys available in cam_sample['cs_record']: {list(cam_sample['cs_record'].keys())}")
            printed_debug_keys = True
        # --- END DEBUG ---

        try:
            img_id = int(img_ids_from_sample[j])
            image = cam_sample['image']
            pc = sample['radar'][j]
            cam_cs_record = cam_sample['cs_record']

            # --- Get token from the cs_record dictionary ---
            sd_token = cam_cs_record.get('token')
            # --------------------------------------------

            if sd_token is None:
                 # If it's still None, log the cs_record keys for further debugging
                 logger.warning(f"Could not find 'token' within cam_sample['cs_record'] for cam {cam_name}, sample {sample.get('sample_idx', 'N/A')}. Skipping camera.")
                 if isinstance(cam_cs_record, dict):
                      logger.warning(f"Keys in cs_record were: {list(cam_cs_record.keys())}")
                 else:
                      logger.warning(f"cs_record was not a dictionary: {type(cam_cs_record)}")
                 continue

        except (IndexError, KeyError, ValueError) as e:
             logger.warning(f"Missing essential data or invalid img_id for camera {cam_name} in sample index {sample.get('sample_idx', 'N/A')}. Skipping camera. Error: {e}")
             continue

        try:
            image = cv2.resize(image, (1600, 900))
        except Exception as e:
            logger.error(f"Failed to resize image for img_id {img_id}, sd_token {sd_token}: {e}")
            continue

        img_height, img_width, _ = image.shape

        # Process annotations
        sample_anns = []
        try:
            annotations = nusc_dataset.pc_to_sensor(sample['annotations'][j], cam_cs_record)
        except IndexError:
             logger.warning(f"Annotation index {j} out of bounds for sample index {sample.get('sample_idx', 'N/A')}. Skipping annotations for this view.")
             annotations = []
        except Exception as e:
             logger.error(f"Error getting/transforming annotations for img_id {img_id}, sd_token {sd_token}: {e}")
             annotations = []

        for ann in annotations:
            coco_cat, coco_cat_id, coco_supercat = nuscene_cat_to_coco(ann.name)
            if coco_cat is None: continue
            cat_id = coco_obj.addCategory(coco_cat, coco_supercat, coco_cat_id)
            try:
                bbox = nuscenes_box_to_coco(ann, np.array(cam_cs_record['camera_intrinsic']), (img_width, img_height))
            except Exception as e: bbox = None
            if bbox is None or not all(val >= 0 for val in bbox): continue
            coco_ann = coco_obj.createAnn(bbox, cat_id)
            sample_anns.append(coco_ann)

        if not sample_anns:
            # logger.debug(f"No valid annotations for img_id {img_id}, sd_token {sd_token}. Skipping image.")
            continue

        # Process point cloud (optional)
        pc_coco = []
        try:
            pc_cam = nusc_dataset.pc_to_sensor(pc, cam_cs_record)
            if pc_cam.shape[1] > 0:
                pc_depth = pc_cam[2, :]
                pc_image_coords = view_points(pc_cam[:3, :], np.array(cam_cs_record['camera_intrinsic']), normalize=True)
                valid_idx = (pc_image_coords[0, :] >= 0) & (pc_image_coords[0, :] < img_width) & \
                            (pc_image_coords[1, :] >= 0) & (pc_image_coords[1, :] < img_height) & \
                            (pc_depth > 0)
                if np.any(valid_idx):
                    pc_image_coords = pc_image_coords[:, valid_idx]
                    pc_depth = pc_depth[valid_idx]
                    pc_coco = np.vstack((pc_image_coords[:2, :], pc_depth))
                    pc_coco = np.transpose(pc_coco).tolist()
        except Exception as e:
            logger.warning(f"Error processing point cloud for img_id {img_id}, sd_token {sd_token}: {e}")
            pc_coco = []

        # Add the sample WITH the NuScenes token
        try:
             coco_obj.addSample(
                 img=image,
                 anns=sample_anns,
                 pointcloud=pc_coco,
                 img_id=img_id,
                 other={'cam_cs_record': cam_cs_record, NUSCENES_TOKEN_KEY: sd_token},
                 img_format='BGR',
                 write_img=True,
             )
             images_added_this_sample += 1 # Increment count *only if* addSample succeeded
        except Exception as e:
            logger.error(f"Failed to add sample for img_id {img_id}, sd_token {sd_token} to COCO object: {e}")
            # Do not increment count if addSample failed

    return images_added_this_sample # Return number of images successfully added

def main():
    args = parse_args()
    logger.info(f"Starting NuScenes to COCO conversion for split: {args.split}")
    logger.info(f"Output directory: {args.out_dir}")
    logger.info(f"Cameras selected: {args.cameras}")
    logger.info(f"NuScenes token will be stored under key: '{NUSCENES_TOKEN_KEY}'")

    if "mini" in args.split: nusc_version = "v1.0-mini"
    elif "test" in args.split: nusc_version = "v1.0-test"
    else: nusc_version = "v1.0-trainval"

    categories_coco = [['person', 'person', 1], ['bicycle', 'vehicle', 2], ['car', 'vehicle', 3],
                       ['motorcycle', 'vehicle', 4], ['bus', 'vehicle', 5], ['truck', 'vehicle', 6]]

    logger.info(f"Loading NuScenes {nusc_version} dataset for split '{args.split}'...")
    try:
        nusc_dataset = NuscenesDataset(nusc_path=args.nusc_root, nusc_version=nusc_version,
                                       split=args.split, coordinates='vehicle',
                                       nsweeps_radar=args.nsweeps_radar,
                                       sensors_to_return=['camera', 'radar'],
                                       pc_mode='camera', logging_level=args.logging_level)
        nusc = nusc_dataset.nusc
        logger.info(f"NuscenesDataset loaded with {len(nusc_dataset)} samples.")
    except Exception as e:
        logger.critical(f"Failed to load NuscenesDataset: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Initializing COCO dataset objects for night, rain, and other splits...")
    split_names = {'night': f"{args.split}_night", 'rain': f"{args.split}_rain",
                   'other': f"{args.split}_not_rain_and_night"}
    coco_splits = {}
    os.makedirs(args.out_dir, exist_ok=True)

    for key, split_name_full in split_names.items():
        coco_obj = COCO_PLUS(logging_level=args.logging_level)
        coco_obj.create_new_dataset(dataset_dir=args.out_dir, split=split_name_full)
        for (coco_cat, coco_supercat, coco_cat_id) in categories_coco:
            coco_obj.addCategory(coco_cat, coco_supercat, coco_cat_id)
        coco_splits[key] = coco_obj
        logger.info(f"Created COCO object for split: {split_name_full}")

    num_samples = len(nusc_dataset)
    logger.info(f"Processing {num_samples} samples from NuScenes dataset...")
    scene_counters = {'night': 0, 'rain': 0, 'other': 0, 'skipped': 0}
    # --- Use len() on the internal lists to track counts ---
    total_images_added = {'night': 0, 'rain': 0, 'other': 0}
    # -------------------------------------------------------

    for i in trange(num_samples, desc="Processing Samples"):
        try:
            sample = nusc_dataset[i]
            sample['sample_idx'] = i # Add index for logging

            if 'scene_token' not in sample:
                 scene_counters['skipped'] += 1
                 continue

            scene_token = sample['scene_token']
            scene = nusc.get('scene', scene_token)
            description = scene['description'].lower()

            assigned_split_key = 'other'
            if 'night' in description: assigned_split_key = 'night'
            elif 'rain' in description: assigned_split_key = 'rain'

            target_coco_obj = coco_splits[assigned_split_key]
            scene_counters[assigned_split_key] += 1 # Count scenes assigned

            # --- Update counts based on the return value of process_and_add_sample ---
            images_added = process_and_add_sample(target_coco_obj, sample, nusc_dataset, args.cameras, categories_coco)
            total_images_added[assigned_split_key] += images_added
            # -----------------------------------------------------------------------

        except KeyboardInterrupt: # Allow stopping gracefully
             logger.warning("KeyboardInterrupt detected. Stopping processing.")
             break
        except Exception as e:
            logger.error(f"Critical error processing sample index {i}: {e}", exc_info=True)
            scene_counters['skipped'] += 1
            continue

    logger.info("Finished processing samples. Saving annotation files...")
    logger.info(f"Scene distribution: Night={scene_counters['night']}, Rain={scene_counters['rain']}, Other={scene_counters['other']}, Skipped={scene_counters['skipped']}")
    logger.info(f"Total images added per split: Night={total_images_added['night']}, Rain={total_images_added['rain']}, Other={total_images_added['other']}")

    for key, coco_obj in coco_splits.items():
        split_name_full = split_names[key]
        # --- Get final counts using len() on the internal dataset dict ---
        final_img_count = len(coco_obj.dataset.get('images', []))
        final_ann_count = len(coco_obj.dataset.get('annotations', []))
        # ---------------------------------------------------------------
        logger.info(f"Saving annotations for split: {split_name_full} ({final_img_count} images, {final_ann_count} annotations)...")
        try:
            coco_obj.saveAnnsToDisk()
            logger.info(f"Successfully saved {split_name_full}.json")
        except Exception as e:
             logger.error(f"Failed to save annotations for split {split_name_full}: {e}")

    logger.info("COCO conversion and splitting complete.")

if __name__ == '__main__':
    main()