import _init_path
import numpy as np
import argparse
import sys
import os
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from det2_port.utils import clip_boxes_to_image
from cocoplus.coco import COCO_PLUS
from rrpn_generator import get_im_proposals
from visualization import draw_xyxy_bbox
from visualization import save_fig
import pickle

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

    args = parser.parse_args()
    return args

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

    # Load the nucoco dataset
    coco = COCO_PLUS(args.ann_file)

    for img_id, img_info in tqdm(coco.imgs.items()):
        img_ind += 1

        if int(args.include_depth)==1:
            proposals = np.empty((0,5), np.float32)
        else:
            proposals = np.empty((0,4), np.float32)

        # Generate proposals for points in pointcloud
        pointcloud = coco.imgToPc[img_id]
        for point in pointcloud['points']:
            rois = get_im_proposals(point, 
                                    sizes=(128, 256, 512, 1024),
                                    aspect_ratios=(0.5, 1, 2),
                                    layout=['center','top','left','right'],
                                    beta=8,
                                    include_depth=args.include_depth)
            proposals = np.append(proposals, np.array(rois), axis=0)
                        
        # Clip the boxes to image boundaries
        img_width = img_info['width']
        img_height = img_info['height']
        proposals = clip_boxes_to_image(proposals, (img_height, img_width))

        boxes.append(proposals)
        scores.append(np.zeros((proposals.shape[0]), dtype=np.float32))
        ids.append(img_id)

    print('Saving proposals to disk...')
    # Save the object using pickle
    with open(output_file, 'wb') as f:
        pickle.dump(dict(boxes=boxes, scores=scores, ids=ids), f)
    print('Proposals saved to {}'.format(output_file))
