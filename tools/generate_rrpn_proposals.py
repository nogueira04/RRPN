import _init_path # noqa: F401 Ensures project modules are found
import numpy as np
import argparse
import os
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from det2_port.utils import clip_boxes_to_image
from cocoplus.coco import COCO_PLUS
from rrpn_generator import get_im_proposals
# Assuming visualization functions are correctly defined elsewhere if needed
# from visualization import draw_xyxy_bbox
# from visualization import save_fig
import pickle
from torchvision.ops import nms  # Import PyTorch's NMS function

# Setup basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Generate object proposals from Radar pointclouds for dataset splits.')

    parser.add_argument('--ann_file_base', required=True,
                        help='Base path structure for annotation files. Keyword will be appended. '
                             'Example: /path/to/data/nucoco/annotations/instances_val -> becomes instances_val_night.json')

    parser.add_argument('--imgs_dir_base', required=True,
                        help='Base path structure for image directories corresponding to splits. '
                              'Example: /path/to/data/nucoco/val -> might need adjustment if images are split too.')

    parser.add_argument('--output_file_base', required=True,
                        help='Base path structure for output proposal files. Keyword will be appended. '
                             'Example: /path/to/data/nucoco/proposals/proposals_val -> becomes proposals_val_night.pkl')

    parser.add_argument('--include_depth', type=int, default=0, choices=[0, 1],
                        help='If 1, include depth information (5th dim) from radar in proposals.')

    parser.add_argument('--nms_threshold', type=float, default=0.8,
                        help='NMS IoU threshold for filtering proposals.')

    parser.add_argument('--keyword_splits', required=True, nargs='+', type=str,
                        help='List of keywords defining the splits to process (e.g., night rain not_rain_and_night).')

    parser.add_argument('--debug', action='store_true',
                        help='Enable debugging to save images with proposals and pointcloud.')

    args = parser.parse_args()

    # Basic validation
    if not args.keyword_splits:
        parser.error("--keyword_splits requires at least one keyword.")

    return args

def calculate_heuristic_scores(proposals, img_width, img_height):
    """
    Calculate normalized scores for each proposal based on area, aspect ratio, and position bias.
    Handles cases where height or width might be zero after clipping.
    """
    if proposals.shape[0] == 0:
        return np.array([], dtype=np.float32)

    x_min, y_min = proposals[:, 0], proposals[:, 1]
    x_max, y_max = proposals[:, 2], proposals[:, 3]
    widths = np.maximum(0., x_max - x_min) # Ensure non-negative
    heights = np.maximum(0., y_max - y_min) # Ensure non-negative
    areas = widths * heights

    image_area = img_width * img_height
    if image_area == 0: # Avoid division by zero
        return np.zeros(proposals.shape[0], dtype=np.float32)

    score_area = areas / image_area

    # Avoid division by zero for aspect ratio, assign low score if height is zero
    aspect_ratios = np.divide(widths, heights, out=np.ones_like(widths), where=heights!=0)
    # Penalize extreme aspect ratios (more than 2:1 or 1:2)
    score_aspect_ratio = 1 - np.abs(np.log2(np.maximum(aspect_ratios, 1e-6))) / np.log2(4) # Penalize ratios > 4 or < 1/4
    score_aspect_ratio = np.clip(score_aspect_ratio, 0, 1)

    img_center_x, img_center_y = img_width / 2, img_height / 2
    box_centers_x = (x_min + x_max) / 2
    box_centers_y = (y_min + y_max) / 2
    distances_to_center = np.sqrt((box_centers_x - img_center_x)**2 + (box_centers_y - img_center_y)**2)
    max_distance = np.sqrt(img_center_x**2 + img_center_y**2)
    if max_distance == 0: # Avoid division by zero for single point image
         score_position = np.ones_like(distances_to_center)
    else:
        score_position = 1 - distances_to_center / max_distance

    # Adjust weighting if desired
    combined_score = (0.5 * score_area + 0.3 * score_aspect_ratio + 0.2 * score_position)
    combined_score = np.clip(combined_score, 0, 1)

    return combined_score

def apply_nms(proposals, scores, nms_threshold):
    """
    Apply NMS to filter proposals based on the NMS threshold.
    Handles empty input.
    """
    if proposals.shape[0] == 0:
        return np.array([], dtype=proposals.dtype).reshape(0, proposals.shape[1]), np.array([], dtype=scores.dtype)

    # Ensure tensors are on CPU for NMS if they came from GPU ops earlier
    proposals_tensor = torch.tensor(proposals[:, :4]).cpu().float() # NMS works on xyxy
    scores_tensor = torch.tensor(scores).cpu().float()

    keep_indices = nms(proposals_tensor, scores_tensor, nms_threshold)

    # Keep only the filtered proposals and scores using original numpy arrays
    filtered_proposals = proposals[keep_indices.numpy()]
    filtered_scores = scores[keep_indices.numpy()]

    return filtered_proposals, filtered_scores

def save_debug_images(img_info, proposals, filtered_proposals, filtered_scores, pointcloud, img_dir, debug_output_dir, split_name):
    """
    Save original image, image with pointcloud, image with proposals before and after filtering.
    Draw filtered proposals with their scores.
    """
    img_id = img_info['id']
    # Construct image path based on base directory and filename from COCO info
    img_file = os.path.join(img_dir, img_info['file_name']) # Assumes file_name is relative path or just basename

    if not os.path.exists(img_file):
        logger.warning(f"Debug image file not found: {img_file}. Skipping debug save for image ID {img_id}.")
        return

    try:
        img = cv2.imread(img_file)
        if img is None:
            logger.warning(f"Failed to load debug image: {img_file}. Skipping debug save for image ID {img_id}.")
            return

        # Create a subdirectory for this split if it doesn't exist
        split_debug_dir = os.path.join(debug_output_dir, split_name)
        os.makedirs(split_debug_dir, exist_ok=True)

        img_copy = img.copy()
        img_proposals_before = img.copy()
        img_proposals_after = img.copy()

        # Save original image
        original_img_file = os.path.join(split_debug_dir, f"{img_id}_original.jpg")
        cv2.imwrite(original_img_file, img)

        # Draw pointcloud on image
        if pointcloud and 'points' in pointcloud:
            for point in pointcloud['points']:
                 if len(point) >= 2: # Ensure point has at least x, y
                    cv2.circle(img_copy, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1) # Green for radar points

            pointcloud_img_file = os.path.join(split_debug_dir, f"{img_id}_pointcloud.jpg")
            cv2.imwrite(pointcloud_img_file, img_copy)

        # Draw proposals before filtering on image
        if proposals is not None and proposals.shape[0] > 0:
            for proposal in proposals:
                x_min, y_min, x_max, y_max = proposal[:4].astype(int)
                cv2.rectangle(img_proposals_before, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2) # Blue

            proposals_before_img_file = os.path.join(split_debug_dir, f"{img_id}_proposals_before.jpg")
            cv2.imwrite(proposals_before_img_file, img_proposals_before)

        # Draw proposals after filtering with scores on image
        if filtered_proposals is not None and filtered_scores is not None and filtered_proposals.shape[0] > 0:
            for proposal, score in zip(filtered_proposals, filtered_scores):
                x_min, y_min, x_max, y_max = proposal[:4].astype(int)
                cv2.rectangle(img_proposals_after, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2) # Red
                # Add score text
                text = f"{score:.2f}"
                text_pos = (x_min, y_min - 10 if y_min > 20 else y_min + 15) # Position above or below box
                cv2.putText(img_proposals_after, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            proposals_after_img_file = os.path.join(split_debug_dir, f"{img_id}_proposals_after.jpg")
            cv2.imwrite(proposals_after_img_file, img_proposals_after)

    except Exception as e:
        logger.error(f"Error saving debug image for ID {img_id}: {e}")


##------------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    logger.info(f"Starting proposal generation for splits: {args.keyword_splits}")

    # Debug output directory setup
    debug_output_dir = None
    if args.debug:
        # Create debug dir relative to the *first* output file base for simplicity
        debug_base_dir = os.path.dirname(args.output_file_base)
        debug_output_dir = os.path.join(debug_base_dir, 'debug_proposal_images')
        os.makedirs(debug_output_dir, exist_ok=True)
        logger.info(f"Debug images will be saved in: {debug_output_dir}")

    # Process each specified keyword split
    for kw in args.keyword_splits:
        logger.info(f"--- Processing keyword split: {kw} ---")

        # --- Initialize lists INSIDE the loop for each split ---
        split_boxes = []
        split_scores = []
        split_ids = []

        # --- Construct filenames using the current keyword 'kw' ---
        # Assumes base name needs _{kw} appended before the extension
        ann_file_kw = f"{args.ann_file_base}_{kw}.json"
        output_file_kw = f"{args.output_file_base}_{kw}.pkl"
        # Determine image directory - adjust if images are also split into folders like val_night, val_rain etc.
        # If images are all in one folder (e.g., 'val'), use args.imgs_dir_base directly.
        # If they are split like annotations, construct the path:
        # img_dir_kw = f"{args.imgs_dir_base}_{kw}" # Example if image dirs are split
        img_dir_kw = args.imgs_dir_base # Assuming images are NOT split into separate folders per keyword

        if not os.path.exists(ann_file_kw):
            logger.warning(f"Annotation file not found for split '{kw}': {ann_file_kw}. Skipping this split.")
            continue

        logger.info(f"Loading annotations from: {ann_file_kw}")
        try:
            coco = COCO_PLUS(ann_file_kw)
        except Exception as e:
             logger.error(f"Failed to load COCO annotations from {ann_file_kw}: {e}")
             continue # Skip to next keyword

        logger.info(f"Generating proposals for {len(coco.imgs)} images in split '{kw}'...")
        # Loop through images defined in *this split's* annotation file
        for img_id, img_info in tqdm(coco.imgs.items(), desc=f"Processing {kw}", unit="image"):

            if int(args.include_depth) == 1:
                proposals = np.empty((0, 5), dtype=np.float32)
            else:
                proposals = np.empty((0, 4), dtype=np.float32)

            # Check if pointcloud data exists for this image
            if img_id not in coco.imgToPc:
                logger.warning(f"No pointcloud data found for image ID {img_id} in split '{kw}'. Generating 0 proposals.")
                pointcloud = None # Ensure pointcloud is None or an empty dict
            else:
                pointcloud = coco.imgToPc[img_id]

            # Generate proposals only if pointcloud exists and has points
            if pointcloud and 'points' in pointcloud and len(pointcloud['points']) > 0:
                # Ensure points are in expected format (at least x, y, possibly depth/features)
                # Assuming rrpn_generator expects points as a list of [x, y, ...]
                valid_points = [p for p in pointcloud['points'] if len(p) >= 2]

                for point in valid_points:
                    try:
                        # Call your proposal generation function
                        rois = get_im_proposals(point,
                                                sizes=(128, 256, 512), # Adjusted example sizes
                                                aspect_ratios=(0.5, 1.0, 2.0), # Adjusted example ratios
                                                layout=['center', 'top', 'bottom', 'left', 'right'], # Adjusted example layout
                                                beta=0.7, # Adjusted example beta
                                                include_depth=(args.include_depth == 1))

                        # Ensure rois is a numpy array before appending
                        if isinstance(rois, list): rois = np.array(rois, dtype=np.float32)

                        # Check shape consistency before appending
                        if rois.ndim == 2 and rois.shape[1] == proposals.shape[1]:
                             proposals = np.append(proposals, rois, axis=0)
                        elif rois.size > 0: # Log if non-empty but wrong shape
                             logger.warning(f"Proposal shape mismatch for img {img_id}. Expected {proposals.shape[1]} dims, got {rois.shape}. Skipping these proposals.")

                    except Exception as e:
                        logger.error(f"Error in get_im_proposals for point {point} in image {img_id}: {e}")
            else:
                # Handle case with no valid points or no pointcloud entry
                if not pointcloud or 'points' not in pointcloud:
                    logger.debug(f"No pointcloud entry for img {img_id}.")
                elif len(pointcloud['points']) == 0:
                     logger.debug(f"Pointcloud for img {img_id} has 0 points.")
                # proposals remains empty array initialized earlier

            # Clip proposals even if empty (no-op)
            img_width = img_info.get('width')
            img_height = img_info.get('height')
            if img_width is None or img_height is None:
                 logger.error(f"Image dimensions missing for img_id {img_id}. Cannot clip or score proposals.")
                 # Append empty arrays if we cannot proceed
                 split_boxes.append(np.empty((0, proposals.shape[1]), dtype=np.float32))
                 split_scores.append(np.empty((0,), dtype=np.float32))
                 split_ids.append(img_id)
                 continue # Skip to next image

            proposals = clip_boxes_to_image(proposals, (img_height, img_width))

            # Calculate heuristic scores
            heuristic_scores = calculate_heuristic_scores(proposals, img_width, img_height)

            # Apply NMS to filter the proposals
            filtered_proposals, filtered_scores = apply_nms(proposals, heuristic_scores, args.nms_threshold)

            # Store the filtered proposals and scores for this image
            split_boxes.append(filtered_proposals)
            split_scores.append(filtered_scores)
            split_ids.append(img_id)

            # --- Save debug images ---
            if args.debug and debug_output_dir:
                # Pass necessary info to the debug function
                save_debug_images(img_info, proposals, filtered_proposals, filtered_scores, pointcloud, img_dir_kw, debug_output_dir, kw)
        # --- End of loop over images for the current split 'kw' ---

        # --- Save proposals for *this specific split* AFTER processing all its images ---
        logger.info(f"Saving proposals for split '{kw}' to disk ({len(split_ids)} images)...")
        output_dir = os.path.dirname(output_file_kw)
        os.makedirs(output_dir, exist_ok=True) # Ensure directory exists

        try:
            with open(output_file_kw, 'wb') as f:
                pickle.dump(dict(boxes=split_boxes, scores=split_scores, ids=split_ids), f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Proposals successfully saved to {output_file_kw}")
        except Exception as e:
            logger.error(f"Failed to save proposals to {output_file_kw}: {e}")

    # --- End of loop over keywords ---
    logger.info("Proposal generation finished for all specified splits.")