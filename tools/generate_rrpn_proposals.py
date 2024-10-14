import _init_path
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
from visualization import draw_xyxy_bbox
from visualization import save_fig
import pickle
from torchvision.ops import nms  # Import PyTorch's NMS function

def parse_args():
    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Generate object proposals from Radar pointclouds.')

    parser.add_argument('--ann_file', dest='ann_file',
                        help='Annotations file',
                        default='../data/nucoco/v1.0-mini/annotations/instances_train.json')

    parser.add_argument('--imgs_dir', dest='imgs_dir',
                        help='Images directory',
                        default='../data/nucoco/v1.0-mini/train')

    parser.add_argument('--out_file', dest='output_file',
                        help='Output filename',
                        default='../data/nucoco/v1.0-mini/proposals/proposals_train.pkl')
    
    parser.add_argument('--include_depth', dest='include_depth',
                        help='If 1, include depth information from radar',
                        default=0)

    parser.add_argument('--nms_threshold', dest='nms_threshold',
                        help='NMS IoU threshold for filtering proposals',
                        type=float, default=0.8)

    parser.add_argument('--debug', dest='debug', 
                        help='Enable debugging to save images with proposals and pointcloud', 
                        action='store_true')

    args = parser.parse_args()
    return args

def calculate_heuristic_scores(proposals, img_width, img_height):
    """
    Calculate normalized scores for each proposal based on area, aspect ratio, and position bias.
    """
    x_min, y_min = proposals[:, 0], proposals[:, 1]
    x_max, y_max = proposals[:, 2], proposals[:, 3]
    widths = x_max - x_min
    heights = y_max - y_min
    areas = widths * heights

    image_area = img_width * img_height
    score_area = areas / image_area

    aspect_ratios = widths / heights
    score_aspect_ratio = 1 - np.abs(1 - aspect_ratios) / np.max([1 - 0.5, 2 - 1])

    img_center_x, img_center_y = img_width / 2, img_height / 2
    box_centers_x = (x_min + x_max) / 2
    box_centers_y = (y_min + y_max) / 2
    distances_to_center = np.sqrt((box_centers_x - img_center_x)**2 + (box_centers_y - img_center_y)**2)
    max_distance = np.sqrt(img_center_x**2 + img_center_y**2)
    score_position = 1 - distances_to_center / max_distance

    combined_score = (0.5 * score_area + 0.3 * score_aspect_ratio + 0.2 * score_position)
    combined_score = np.clip(combined_score, 0, 1) 

    return combined_score

def apply_nms(proposals, scores, nms_threshold):
    """
    Apply NMS to filter proposals based on the NMS threshold.
    """
    proposals_tensor = torch.tensor(proposals)  # Convert proposals to a tensor
    scores_tensor = torch.tensor(scores)        # Convert scores to a tensor
    
    # Perform NMS: returns the indices of the proposals to keep
    keep_indices = nms(proposals_tensor, scores_tensor, nms_threshold)
    
    # Keep only the filtered proposals and scores
    filtered_proposals = proposals_tensor[keep_indices].numpy()
    filtered_scores = scores_tensor[keep_indices].numpy()
    
    return filtered_proposals, filtered_scores

def save_debug_images(img_info, proposals, filtered_proposals, filtered_scores, pointcloud, coco, args, output_dir):
    """
    Save original image, image with pointcloud, image with proposals before and after filtering.
    Draw filtered proposals with their scores.
    """
    img_id = img_info['id']
    img_file = os.path.join(args.imgs_dir, img_info['file_name'])
    
    # Load image
    img = cv2.imread(img_file)
    img_copy = img.copy()
    img_proposals_before = img.copy()
    img_proposals_after = img.copy()

    # Save original image
    original_img_file = os.path.join(output_dir, f"{img_id}_original.jpg")
    cv2.imwrite(original_img_file, img)

    # Draw pointcloud on image
    for point in pointcloud['points']:
        cv2.circle(img_copy, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Green for radar points
    
    pointcloud_img_file = os.path.join(output_dir, f"{img_id}_pointcloud.jpg")
    cv2.imwrite(pointcloud_img_file, img_copy)

    # Draw proposals before filtering on image
    if proposals is not None:
        proposals = np.array(proposals)  # Ensure proposals is a NumPy array
        for proposal in proposals:
            x_min, y_min, x_max, y_max = proposal[:4].astype(int)
            cv2.rectangle(img_proposals_before, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue for proposals before filtering

        proposals_before_img_file = os.path.join(output_dir, f"{img_id}_proposals_before.jpg")
        cv2.imwrite(proposals_before_img_file, img_proposals_before)

    # Draw proposals after filtering with scores on image
    if filtered_proposals is not None and filtered_scores is not None:
        filtered_proposals = np.array(filtered_proposals)  # Ensure filtered_proposals is a NumPy array
        filtered_scores = np.array(filtered_scores)      # Ensure filtered_scores is a NumPy array
        for proposal, score in zip(filtered_proposals, filtered_scores):
            x_min, y_min, x_max, y_max = proposal[:4].astype(int)
            cv2.rectangle(img_proposals_after, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red for filtered proposals
            cv2.putText(img_proposals_after, f"{score:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # Display the score

        proposals_after_img_file = os.path.join(output_dir, f"{img_id}_proposals_after.jpg")
        cv2.imwrite(proposals_after_img_file, img_proposals_after)




##------------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    output_file = args.output_file
    boxes = []
    scores = []
    ids = []
    img_ind = 0

    out_dir = os.path.dirname(output_file)
    os.makedirs(out_dir, exist_ok=True)

    # Debug output directory
    if args.debug:
        debug_output_dir = os.path.join(out_dir, 'debug_images')
        os.makedirs(debug_output_dir, exist_ok=True)

    # Load the nucoco dataset
    coco = COCO_PLUS(args.ann_file)

    for img_id, img_info in tqdm(coco.imgs.items()):
        img_ind += 1

        if int(args.include_depth) == 1:
            proposals = np.empty((0, 5), np.float32)
        else:
            proposals = np.empty((0, 4), np.float32)

        # Generate proposals for points in pointcloud
        pointcloud = coco.imgToPc[img_id]
        for point in pointcloud['points']:
            rois = get_im_proposals(point, 
                                    sizes=(128, 256, 512, 1024),
                                    aspect_ratios=(0.5, 1),
                                    layout=['center', 'top', 'left', 'right'],
                                    beta=8,
                                    include_depth=args.include_depth)
            proposals = np.append(proposals, np.array(rois), axis=0)

        # Clip the boxes to image boundaries
        img_width = img_info['width']
        img_height = img_info['height']
        proposals = clip_boxes_to_image(proposals, (img_height, img_width))

        # Calculate heuristic scores
        heuristic_scores = calculate_heuristic_scores(proposals, img_width, img_height)

        # Save debug images before filtering
        if args.debug:
            save_debug_images(img_info, proposals, None, None, pointcloud, coco, args, debug_output_dir)

        # Apply NMS to filter the proposals
        filtered_proposals, filtered_scores = apply_nms(proposals, heuristic_scores, args.nms_threshold)

        # Store the filtered proposals and scores
        boxes.append(filtered_proposals)
        scores.append(filtered_scores)
        ids.append(img_id)

        # Save debug images after filtering
        if args.debug:
            save_debug_images(img_info, proposals, filtered_proposals, filtered_scores, pointcloud, coco, args, debug_output_dir)

    print('Saving proposals to disk...')
    # Save the object using pickle
    with open(output_file, 'wb') as f:
        pickle.dump(dict(boxes=boxes, scores=scores, ids=ids), f)
    print('Proposals saved to {}'.format(output_file))
