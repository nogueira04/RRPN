# File: nuscenes_to_coco_combined_final_v3.py
# Purpose: Converts a NuScenes split into a single combined COCO dataset,
#          storing the NuScenes token AND creating a separate mapping file
#          from the generated COCO image ID -> NuScenes scene type.

import os
import sys
import numpy as np
import argparse
import cv2
from tqdm import tqdm, trange
import json # Need json import
import pickle # Need pickle to save the mapping

try:
    from cocoplus.coco import COCO_PLUS
except ImportError:
    print("ERROR: Failed to import COCO_PLUS. Make sure it's installed or accessible.")
    sys.exit(1)
from pynuscenes.utils.nuscenes_utils import nuscenes_box_to_coco, nuscene_cat_to_coco
from pynuscenes.nuscenes_dataset import NuscenesDataset
from nuscenes.utils.geometry_utils import view_points
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Define the key name used to store the token in the COCO JSON ---
NUSCENES_TOKEN_KEY = "nusc_token"
# ---

def parse_args():
    # Add argument for the mapping file output path
    parser = argparse.ArgumentParser(description='Converts NuScenes split to combined COCO format AND creates an image_id->scene_type map.')
    parser.add_argument('--nusc_root', required=True, help='NuScenes dataroot path.')
    parser.add_argument('--split', required=True, choices=['val', 'train', 'mini_train', 'mini_val'], help='NuScenes dataset split.')
    parser.add_argument('--out_dir', required=True, help='Output base directory for COCO dataset.')
    # --- New Argument for Mapping File ---
    parser.add_argument('--map_out_file', required=True, help='Output path for the image_id -> scene_type mapping pickle file (e.g., ../data/nucoco/annotations/id_to_scene_val.pkl).')
    # ------------------------------------
    parser.add_argument('--nsweeps_radar', default=1, type=int, help='Number of Radar sweeps.')
    parser.add_argument('--cameras', nargs='+', default=['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'], help='List of cameras.')
    parser.add_argument('-l', '--logging_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging level.')
    args = parser.parse_args()
    logging.getLogger().setLevel(args.logging_level)
    logger.setLevel(args.logging_level)
    return args

def main():
    args = parse_args()
    logger.info(f"Starting NuScenes to COMBINED COCO conversion for split: {args.split}")
    logger.info(f"Output directory: {args.out_dir}")
    logger.info(f"Scene type mapping will be saved to: {args.map_out_file}")
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
        nusc = nusc_dataset.nusc # Get the underlying NuScenes API object
        logger.info(f"NuscenesDataset loaded with {len(nusc_dataset)} samples.")
    except Exception as e:
        logger.critical(f"Failed to load NuscenesDataset: {e}", exc_info=True)
        sys.exit(1)

    # Initialize COCO object
    logger.info(f"Initializing COCO dataset object for the combined '{args.split}' split...")
    combined_coco_obj = COCO_PLUS(logging_level=args.logging_level)
    combined_coco_obj.create_new_dataset(dataset_dir=args.out_dir, split=args.split)
    for (coco_cat, coco_supercat, coco_cat_id) in categories_coco:
        combined_coco_obj.addCategory(coco_cat, coco_supercat, coco_cat_id)
    logger.info(f"Created COCO object for combined split: {args.split}")

    # --- Initialize mapping dictionary ---
    image_id_to_scene_type = {}
    # -----------------------------------

    num_samples = len(nusc_dataset)
    logger.info(f"Processing {num_samples} samples from NuScenes dataset...")
    total_images_added = 0
    skipped_samples_count = 0
    skipped_cameras_count = 0
    failed_bbox_count = 0
    failed_addsample_count = 0
    img_ids_added = set() # Track added IDs to prevent duplicates

    for i in trange(num_samples, desc="Processing Samples"):
        try:
            sample = nusc_dataset[i]
            sample['sample_idx'] = i
            img_ids = sample.get('img_id')

            if img_ids is None:
                logger.warning(f"Sample {i} missing 'img_id'. Skipping.")
                skipped_samples_count += 1
                continue

            # --- Determine Scene Type ONCE per NuScenes Sample ---
            scene_token = sample.get('scene_token')
            if not scene_token:
                 logger.warning(f"Sample {i} missing 'scene_token'. Cannot determine scene type. Skipping.")
                 skipped_samples_count += 1
                 continue
            try:
                scene_record = nusc.get('scene', scene_token)
                description = scene_record['description'].lower()
                scene_type = 'other'
                if 'night' in description: scene_type = 'night'
                elif 'rain' in description: scene_type = 'rain'
            except KeyError:
                logger.warning(f"Scene token '{scene_token}' from sample {i} not found in NuScenes DB. Skipping sample.")
                skipped_samples_count += 1
                continue
            # ---------------------------------------------------

        except Exception as e:
            logger.error(f"Error retrieving sample index {i} or its scene: {e}", exc_info=True)
            skipped_samples_count += 1
            continue

        # Iterate through camera views
        for j, cam_sample in enumerate(sample.get('camera', [])):
            cam_name = cam_sample.get('camera_name')
            if not cam_name or cam_name not in args.cameras: continue

            try:
                img_id = int(img_ids[j]) # Get the COCO ID for this view

                # --- Prevent adding the same img_id twice ---
                if img_id in img_ids_added:
                    logger.warning(f"Attempting to re-add img_id {img_id} (Sample {i}, Cam {cam_name}). Skipping this view. Check pynuscenes ID generation.")
                    skipped_cameras_count += 1
                    continue
                # --------------------------------------------

                image = cam_sample['image']
                pc = sample['radar'][j]
                cam_cs_record = cam_sample['cs_record']
                sd_token = cam_cs_record.get('token') # Still useful to save token

                if sd_token is None:
                    logger.warning(f"Could not find 'token' within cam_cs_record for cam {cam_name}, sample {i}, img_id {img_id}. Skipping camera view.")
                    skipped_cameras_count += 1
                    continue

                # ... (Image Preprocessing) ...
                image = cv2.resize(image, (1600, 900))
                img_height, img_width, _ = image.shape

                # ... (Annotation Processing & Filtering) ...
                sample_anns = []
                annotations = nusc_dataset.pc_to_sensor(sample['annotations'][j], cam_cs_record)
                for ann in annotations:
                    coco_cat, coco_cat_id, coco_supercat = nuscene_cat_to_coco(ann.name)
                    if coco_cat is None: continue
                    cat_id = combined_coco_obj.addCategory(coco_cat, coco_supercat, coco_cat_id)
                    try:
                        bbox = nuscenes_box_to_coco(ann, np.array(cam_cs_record['camera_intrinsic']), (img_width, img_height))
                    except Exception: bbox = None
                    if bbox is None or not all(val >= 0 for val in bbox):
                        failed_bbox_count += 1
                        continue
                    coco_ann = combined_coco_obj.createAnn(bbox, cat_id)
                    sample_anns.append(coco_ann)

                # --- Filter Image based on Annotations ---
                if not sample_anns:
                    skipped_cameras_count += 1
                    continue
                # --------------------------------------

                # ... (Point Cloud Processing - Optional) ...
                pc_coco = []
                # ... (rest of point cloud logic) ...

                # --- Add Sample to COCO Dataset ---
                try:
                    img_format_to_pass = 'BGR' # Assuming BGR from pynuscenes
                    combined_coco_obj.addSample(
                        img=image,
                        anns=sample_anns,
                        pointcloud=pc_coco,
                        img_id=img_id,
                        other={'cam_cs_record': cam_cs_record, NUSCENES_TOKEN_KEY: sd_token},
                        img_format=img_format_to_pass,
                        write_img=True
                    )
                    # --- Add to mapping *after* successful addSample ---
                    image_id_to_scene_type[img_id] = scene_type
                    img_ids_added.add(img_id) # Mark ID as added
                    total_images_added += 1
                    # -------------------------------------------------
                except Exception as e:
                    logger.error(f"Failed to add sample for img_id {img_id}, token {sd_token}: {e}", exc_info=True)
                    failed_addsample_count += 1
                # ---------------------------------

            except IndexError as e:
                 logger.warning(f"IndexError likely accessing img_ids/annotations/radar for cam index {j} in sample {i}. Skipping view. Error: {e}")
                 skipped_cameras_count += 1
                 continue
            except Exception as e:
                logger.error(f"Unexpected error processing camera index {j} in sample {i}: {e}", exc_info=True)
                skipped_cameras_count += 1
                continue

    # ... (Final logging of counts before saving) ...
    logger.info(f"Total images added: {total_images_added} (Unique IDs added: {len(img_ids_added)})")
    # ... (Log other skipped/failed counts) ...


    # --- Save the COCO annotation file ---
    final_img_count = len(combined_coco_obj.dataset.get('images', []))
    final_ann_count = len(combined_coco_obj.dataset.get('annotations', []))
    logger.info(f"Saving COCO annotations: {args.split} ({final_img_count} images, {final_ann_count} annotations)...")
    try:
        combined_coco_obj.saveAnnsToDisk()
        logger.info(f"Successfully saved COCO annotations to {combined_coco_obj.annotation_file}")
    except Exception as e:
         logger.error(f"Failed to save COCO annotations: {e}")
    # -----------------------------------

    # --- Save the ID -> Scene Type Mapping ---
    logger.info(f"Saving image_id -> scene_type map ({len(image_id_to_scene_type)} entries) to {args.map_out_file}...")
    try:
        # Ensure output directory for map file exists
        map_dir = os.path.dirname(args.map_out_file)
        if map_dir: # Check if path includes a directory
             os.makedirs(map_dir, exist_ok=True)
        with open(args.map_out_file, 'wb') as f:
            pickle.dump(image_id_to_scene_type, f)
        logger.info("Successfully saved mapping file.")
    except Exception as e:
        logger.error(f"Failed to save mapping file to {args.map_out_file}: {e}")
    # ---------------------------------------

    logger.info("COMBINED COCO conversion and mapping generation complete.")

if __name__ == '__main__':
    main()