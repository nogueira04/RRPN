import argparse
import os
import sys
import time
import torch
import pickle
import cv2
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.structures import Boxes, Instances
from detectron2.utils.visualizer import Visualizer

# Dataset registration
_DATASETS = {
    'nucoco_val': {
        'img_dir': '/clusterlivenfs/gnmp/RRPN/data/nucoco/val',
        'ann_file': '/clusterlivenfs/gnmp/RRPN/data/nucoco/annotations/instances_val.json',
    },
    'nucoco_train': {
        'img_dir': '/clusterlivenfs/gnmp/RRPN/data/nucoco/train',
        'ann_file': '/clusterlivenfs/gnmp/RRPN/data/nucoco/annotations/instances_train.json',
    },
}

category_id_to_name = {0: "_", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "bus", 6: "truck"}

def register_datasets():
    for dataset_name, dataset_info in _DATASETS.items():
        # Register the dataset
        DatasetCatalog.register(dataset_name, lambda dataset_info=dataset_info: load_coco_json(dataset_info['ann_file'], dataset_info['img_dir']))
        # Set the metadata for the dataset (e.g., class names)
        MetadataCatalog.get(dataset_name).set(thing_classes=list(category_id_to_name.values()))

DatasetCatalog.clear()  # Clear any existing dataset registration
MetadataCatalog.clear()  # Clear existing metadata
register_datasets()

# Load proposals from file
def load_proposals(proposal_file):
    with open(proposal_file, 'rb') as f:
        return pickle.load(f)

# Map image ids to proposals for faster lookup
def get_proposals(image_id, proposals):
    proposal_ids = set(proposals['ids'])
    id_to_index = {img_id: idx for idx, img_id in enumerate(proposals['ids'])}
    if image_id in proposal_ids:
        idx = id_to_index[image_id]
        return proposals['boxes'][idx], proposals['scores'][idx]
    else:
        raise ValueError(f"No proposals found for image_id {image_id}")

# Inference function
def perform_inference(image_path, image_id, model, proposals):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")
    
    proposal_boxes, proposal_scores = get_proposals(image_id, proposals)
    
    # Prepare proposals for Detectron2
    instances = Instances(image.shape[:2])
    instances.proposal_boxes = Boxes(torch.tensor(proposal_boxes))
    instances.scores = torch.tensor(proposal_scores)
    
    # Convert image to tensor and prepare inputs
    image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image_tensor, "proposals": instances}]
    
    # Perform inference
    with torch.no_grad():
        outputs = model(inputs)[0]
    
    return outputs, image, proposal_boxes

def main(args):
    # Load configuration and model
    cfg = get_cfg()
    if args.cfg_file and os.path.exists(args.cfg_file):
        cfg.set_new_allowed(True)
        cfg.merge_from_file(args.cfg_file)
    else:
        raise ValueError(f"Config file {args.cfg_file} not found.")
    
    # Set model weights
    cfg.MODEL.WEIGHTS = args.model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Build and load model
    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(args.model_weights)
    print(cfg)
    # Load proposals
    proposals = load_proposals("/clusterlivenfs/gnmp/RRPN/data/nucoco/proposals/proposals_val.pkl")

    # Create output directory if not exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Process the dataset
    dataset_dicts = DatasetCatalog.get("nucoco_val")
    for d in dataset_dicts:
        image_path = d["file_name"]
        image_id = int(os.path.splitext(os.path.basename(image_path))[0])
        
        # Perform inference
        outputs, image, proposal_boxes = perform_inference(image_path, image_id, model, proposals)
        
        # # Save original image
        # original_image_path = os.path.join(args.output_dir, f"original_{os.path.basename(image_path)}")
        # cv2.imwrite(original_image_path, image)
        
        # # Visualize proposals
        # v_proposals = Visualizer(image[:, :, ::-1], MetadataCatalog.get("nucoco_val"), scale=1.2)
        # v_proposals = v_proposals.overlay_instances(boxes=proposal_boxes)
        # proposal_image = v_proposals.get_image()[:, :, ::-1]
        # proposals_save_path = os.path.join(args.output_dir, f"proposals_{os.path.basename(image_path)}")
        # cv2.imwrite(proposals_save_path, proposal_image)
        
        # Visualize predictions
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("nucoco_val"), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        result = v.get_image()[:, :, ::-1]
        result_save_path = os.path.join(args.output_dir, f"predictions_{os.path.basename(image_path)}")
        cv2.imwrite(result_save_path, result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file path', required=True)
    parser.add_argument('--model-weights', dest='model_weights', help='Model weights path', required=True)
    parser.add_argument('--output-dir', dest='output_dir', help='Directory to save output images', required=True)
    args = parser.parse_args()
    main(args)
