import argparse
import os
import sys
import time
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import load_coco_json
import cv2

# Dataset registration
_DATASETS = {
    'nucoco_mini_val': {
        'img_dir': '/home/live/RRPNv2/RRPN/data/nucoco/mini_val',
        'ann_file': '/home/live/RRPNv2/RRPN/data/nucoco/annotations/instances_mini_val.json',
    },
    'nucoco_mini_train': {
        'img_dir': '/home/live/RRPNv2/RRPN/data/nucoco/mini_train',
        'ann_file': '/home/live/RRPNv2/RRPN/data/nucoco/annotations/instances_mini_train.json',
    },
}

category_id_to_name = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "bus", 5: "truck"}

def register_datasets():
    for dataset_name, dataset_info in _DATASETS.items():
        # Register the dataset
        DatasetCatalog.register(dataset_name, lambda dataset_info=dataset_info: load_coco_json(dataset_info['ann_file'], dataset_info['img_dir']))
        # Set the metadata for the dataset (e.g., class names)
        MetadataCatalog.get(dataset_name).set(thing_classes=list(category_id_to_name.values()))

register_datasets()

def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--model-weights',
        dest='model_weights',
        help='path to model weights',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory to save output images',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wait',
        dest='wait',
        help='wait until net file exists',
        default=True,
        type=bool
    )
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true'
    )
    parser.add_argument(
        '--multi-gpu-testing',
        dest='multi_gpu_testing',
        help='using cfg.NUM_GPUS for inference',
        action='store_true'
    )
    parser.add_argument(
        '--range',
        dest='range',
        help='start (inclusive) and end (exclusive) indices',
        default=None,
        type=int,
        nargs=2
    )
    parser.add_argument(
        'opts',
        help='See detectron2/config/defaults.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def main(args):
    cfg = get_cfg()

    # Create a config file if it doesn't exist
    if not os.path.exists(args.cfg_file):
        with open(args.cfg_file, 'w') as f:
            f.write(cfg.dump())
            print(f"Created config file at: {args.cfg_file}")

    cfg.merge_from_file(args.cfg_file)
    print(f"Loaded configuration:\n{cfg}")
    cfg.MODEL.WEIGHTS = args.model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set the testing threshold for this model
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.1
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(category_id_to_name)
    cfg.DATASETS.TRAIN = ("nucoco_mini_train", )
    cfg.DATASETS.TEST = ("nucoco_mini_val", )

    if not cfg.DATASETS.TRAIN:
        print("Warning: cfg.DATASETS.TRAIN is empty. Setting it to nucoco_mini_val.")
        cfg.DATASETS.TRAIN = ("nucoco_mini_val", )

    predictor = DefaultPredictor(cfg)
    print("Created predictor.")

    dataset_dicts = DatasetCatalog.get("nucoco_mini_val")
    for d in dataset_dicts:
        print(f"Processing image: {d['file_name']}")
        im = cv2.imread(d["file_name"])
        if im is None:
            print(f"Failed to load image: {d['file_name']}")
            continue
        outputs = predictor(im)
        print(f"Made predictions: {outputs}")

        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("nucoco_mini_val"), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        print("Drew predictions.")
        result = v.get_image()[:, :, ::-1]

        # Resize the image
        max_dim = 800  # Maximum dimension for display
        height, width = result.shape[:2]
        scale = max_dim / max(height, width)
        result = cv2.resize(result, None, fx=scale, fy=scale)

        # Display the image
        cv2.imshow('Inference', result)
        # Wait for a key press and close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # cv2.imwrite(os.path.join(args.output_dir, os.path.basename(d["file_name"])), result)
        # print(f"Saved image to: {os.path.join(args.output_dir, os.path.basename(d['file_name'])}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
