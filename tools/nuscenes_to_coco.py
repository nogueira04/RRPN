import _init_path
import os
import sys
import pickle
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm, trange
from cocoplus.coco import COCO_PLUS
from pynuscenes.utils.nuscenes_utils import nuscenes_box_to_coco, nuscene_cat_to_coco
from pynuscenes.nuscenes_dataset import NuscenesDataset
from nuscenes.utils.geometry_utils import view_points

def parse_args():
    parser = argparse.ArgumentParser(description='Converts the NuScenes dataset to COCO format')
    
    parser.add_argument('--nusc_root', default='../data/nuscenes',
                        help='NuScenes dataroot')
    
    parser.add_argument('--split', default='mini_train',
                        help='Dataset split (mini_train, mini_val, train, val, test)')

    parser.add_argument('--out_dir', default='../data/nucoco/',
                        help='Output directory for the nucoco dataset')

    parser.add_argument('--nsweeps_radar', default=1, type=int,
                        help='Number of Radar sweeps to include')

    parser.add_argument('--use_symlinks', type=bool, default=False,
                        help='Create symlinks to nuScenes images rather than copying them')

    parser.add_argument('--cameras', nargs='+',
                        default=['CAM_FRONT',
                                 'CAM_BACK',
                                #  'CAM_FRONT_LEFT',
                                #  'CAM_FRONT_RIGHT',
                                #  'CAM_BACK_LEFT',
                                #  'CAM_BACK_RIGHT',
                                 ],
                        help='List of cameras to use.')
    
    parser.add_argument('-l', '--logging_level', default='INFO',
                        help='Logging level')
                        
    args = parser.parse_args()
    return args

def showImgAnn(image, annotations, bbox_only=False, BGR=True, save_path=None):
    if BGR:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for ann in annotations:
        bbox = ann['bbox']
        if bbox_only:
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        else:
            pass

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def main():
    args = parse_args()
    print(f"Parsed arguments: {args}")

    if "mini" in args.split:
        nusc_version = "v1.0-mini"
    elif "test" in args.split:
        nusc_version = "v1.0-test"
    else:
        nusc_version = "v1.0-trainval"

    categories = [['person',      'person' ,  1],
                  ['bicycle',     'vehicle',  2],
                  ['car',         'vehicle',  3],
                  ['motorcycle',  'vehicle',  4],
                  ['bus',         'vehicle',  5],
                  ['truck',       'vehicle',  6]
    ]
    
    anns_file = os.path.join(args.out_dir, 'annotations', 'instances_' + args.split + '.json')
    print("Creating COCO dataset for split: {}".format(nusc_version))
    nusc_dataset = NuscenesDataset(nusc_path=args.nusc_root, 
                                   nusc_version=nusc_version, 
                                   split=args.split,
                                   coordinates='vehicle',
                                   nsweeps_radar=args.nsweeps_radar, 
                                   sensors_to_return=['camera', 'radar'],
                                   pc_mode='camera',
                                   logging_level=args.logging_level)
    
    coco_dataset = COCO_PLUS(logging_level="INFO")
    coco_dataset.create_new_dataset(dataset_dir=args.out_dir, split=args.split)

    for (coco_cat, coco_supercat, coco_cat_id) in categories:
        coco_dataset.addCategory(coco_cat, coco_supercat, coco_cat_id)
    
    num_samples = len(nusc_dataset)
    for i in trange(num_samples):
        sample = nusc_dataset[i]
        img_ids = sample['img_id']

        for j, cam_sample in enumerate(sample['camera']):
            if cam_sample['camera_name'] not in args.cameras:
                continue

            img_id = int(img_ids[j])
            image = cam_sample['image']
            pc = sample['radar'][j]
            cam_cs_record = cam_sample['cs_record']

            image = cv2.resize(image, (1600, 900))
            img_height, img_width, _ = image.shape

            sample_anns = []
            annotations = nusc_dataset.pc_to_sensor(sample['annotations'][j], 
                                                    cam_cs_record)

            for ann in annotations:
                print(f"Annotation before conversion: {ann}")

                coco_cat, coco_cat_id, coco_supercat = nuscene_cat_to_coco(ann.name)
                if coco_cat is None:
                    coco_dataset.logger.debug('Skipping ann with category: {}'.format(ann.name))
                    continue

                cat_id = coco_dataset.addCategory(coco_cat, coco_supercat, coco_cat_id)

                print(f"Annotation: {ann}")
                print(f"Camera intrinsic: {cam_cs_record['camera_intrinsic']}")
                print(f"Image dimensions: {(img_width, img_height)}")

                try:
                    bbox = nuscenes_box_to_coco(ann, np.array(cam_cs_record['camera_intrinsic']), 
                                                (img_width, img_height))
                except Exception as e:
                    print(f"Exception while computing bbox: {e}")
                    bbox = None

                if bbox is None:
                    print(f"Failed to compute bbox for annotation: {ann}")
                    continue

                coco_ann = coco_dataset.createAnn(bbox, cat_id)
                sample_anns.append(coco_ann)

            pc_cam = nusc_dataset.pc_to_sensor(pc, cam_cs_record)
            pc_depth = pc_cam[2, :]
            pc_image = view_points(pc_cam[:3, :], 
                                 np.array(cam_cs_record['camera_intrinsic']), 
                                 normalize=True)
            
            pc_coco = np.vstack((pc_image[:2,:], pc_depth))
            pc_coco = np.transpose(pc_coco).tolist()

            print(f"Saving image {img_id} to COCO dataset")
            coco_img_path = coco_dataset.addSample(img=image,
                                                   anns=sample_anns, 
                                                   pointcloud=pc_coco,
                                                   img_id=img_id,
                                                   other=cam_cs_record,
                                                   img_format='RGB',
                                                   write_img=True)
            
            showImgAnn(np.asarray(image), sample_anns, bbox_only=True, BGR=False, save_path=f"visualization_{img_id}.png")
        
    coco_dataset.saveAnnsToDisk()

if __name__ == '__main__':
    main()
