import argparse
import os
import sys
import time
import torch
import pickle
import cv2
import yaml
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.structures import Boxes, Instances
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from timer import Timer

current_path = os.path.dirname(os.path.abspath(__file__))

while os.path.basename(current_path) != 'RRPN':
    parent = os.path.dirname(current_path)
    if parent == current_path:
        raise RuntimeError("Couldn't find 'RRPN' in path hierarchy.")
    current_path = parent

root_dir = current_path

dataset_config_path = os.path.join(root_dir, "configs/general_config.yaml")
dataset_config_path = "/clusterlivenfs/gnmp/RRPN/configs/general_config.yaml"
dataset_config = None
with open(dataset_config_path, "r") as f:
    dataset_config = yaml.safe_load(f)

val_config = dataset_config["NUCOCO"]["VAL"]
train_config = dataset_config["NUCOCO"]["TRAIN"]

timer = Timer()

# Dataset registration
_DATASETS = {
    'nucoco_val': {
            'img_dir': val_config["IMG_DIR"],
            'ann_file': val_config["ANNOT_DIR"],
    },
    'nucoco_train': {
        'img_dir': train_config["IMG_DIR"],
        'ann_file': train_config["ANNOT_DIR"],
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

def plot_timing_graph(timing_summary, output_dir):
    steps = list(timing_summary.keys())
    mean_times = [sum(timing_summary[step]) / len(timing_summary[step]) for step in steps]
    total_times = [sum(timing_summary[step]) for step in steps]

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    x = range(len(steps))

    bars1 = ax.bar(x, mean_times, width, label="Mean Time (ms)", color='b', alpha=0.7)

    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Pipeline Steps")
    ax.set_title("Execution Time per Step (Across All Images)")
    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=45, ha="right")
    ax.legend()

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color='blue')

    plt.tight_layout()
    
    graph_path = os.path.join(output_dir, "timing_graph.png")
    plt.savefig(graph_path)
    print(f"Saved timing graph to {graph_path}")
    plt.close()


# Inference function
def perform_inference(image_path, image_id, model, proposals):
    with timer.time("Loading image"):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        
    with timer.time("Loading proposals"):    
        proposal_boxes, proposal_scores = get_proposals(image_id, proposals)
    
    with timer.time("Prepare proposals for Detectron2"):
        # Prepare proposals for Detectron2
        instances = Instances(image.shape[:2])
        instances.proposal_boxes = Boxes(torch.tensor(proposal_boxes))
        instances.scores = torch.tensor(proposal_scores)
    
    with timer.time("Prepare inputs"):
        # Convert image to tensor and prepare inputs
        image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image_tensor, "proposals": instances}]
    
    with timer.time("Perform inference"):
        # Perform inference
        with torch.no_grad():
            outputs = model(inputs)[0]
        
    return outputs, image, proposal_boxes

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

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
    cfg.DATASETS.TEST = ("nucoco_val",)
    cfg.DATASETS.PROPOSAL_FILES_TEST = ("/clusterlivenfs/gnmp/RRPN/data/nucoco/proposals/proposals_val_rain.pkl",)

    # Build and load model
    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(args.model_weights)
    print(cfg)
    proposals = load_proposals("/clusterlivenfs/gnmp/RRPN/data/nucoco/proposals/proposals_val_rain.pkl")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    timing_summary = {}

    dataset_dicts = DatasetCatalog.get("nucoco_val")
    i = 0
    for d in dataset_dicts:
        if i == 30:
            break
        i += 1
        image_path = d["file_name"]
        image_id = int(os.path.splitext(os.path.basename(image_path))[0])
        
        outputs, image, proposal_boxes = perform_inference(image_path, image_id, model, proposals)

        for step in timer.timings:  
            if step not in timing_summary:  
                timing_summary[step] = []   
            timing_summary[step].extend(timer.timings[step])  # Append times before clearing    

        timer.print_summary()
        timer.timings.clear()
        
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("nucoco_val"), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        result = v.get_image()[:, :, ::-1]
        result_save_path = os.path.join(args.output_dir, f"predictions_{os.path.basename(image_path)}")
        cv2.imwrite(result_save_path, result)

        if args.debug:
            debug_dir = os.path.join(args.output_dir, 'debug', f"{image_id}")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Ground truth
            v_gt = Visualizer(image[:, :, ::-1], MetadataCatalog.get("nucoco_val"), scale=1.2)
            v_gt = v_gt.draw_dataset_dict(d)
            result_gt = v_gt.get_image()[:, :, ::-1]
            cv2.imwrite(os.path.join(debug_dir, "ground_truth.jpg"), result_gt)
            
            # Inference
            v_inference = Visualizer(image[:, :, ::-1], MetadataCatalog.get("nucoco_val"), scale=1.2)
            v_inference = v_inference.draw_instance_predictions(outputs["instances"].to("cpu"))
            result_inference = v_inference.get_image()[:, :, ::-1]
            cv2.imwrite(os.path.join(debug_dir, "inference.jpg"), result_inference)

            gt_boxes = []
            for ann in d["annotations"]:
                x, y, w, h = ann["bbox"]
                gt_boxes.append([x, y, x + w, y + h])

            pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

            iou_threshold = 0.5
            fn_found = False
            fp_found = False

            for g in gt_boxes:
                if max([compute_iou(g, p) for p in pred_boxes] or [0]) < iou_threshold:
                    fn_found = True
                    break

            for p in pred_boxes:
                if max([compute_iou(p, g) for g in gt_boxes] or [0]) < iou_threshold:
                    fp_found = True
                    break

            if fn_found:
                debug_dir_fn = os.path.join(args.output_dir, 'debug_false_negative', str(image_id))
                os.makedirs(debug_dir_fn, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir_fn, "ground_truth.jpg"), result_gt)
                cv2.imwrite(os.path.join(debug_dir_fn, "inference.jpg"), result_inference)
                debug_dir_partial = os.path.join(args.output_dir, 'debug_not_all_found', str(image_id))
                os.makedirs(debug_dir_partial, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir_partial, "ground_truth.jpg"), result_gt)
                cv2.imwrite(os.path.join(debug_dir_partial, "inference.jpg"), result_inference)

            if fp_found:
                debug_dir_fp = os.path.join(args.output_dir, 'debug_false_positive', str(image_id))
                os.makedirs(debug_dir_fp, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir_fp, "ground_truth.jpg"), result_gt)
                cv2.imwrite(os.path.join(debug_dir_fp, "inference.jpg"), result_inference)

    plot_timing_graph(timing_summary, args.output_dir)
    evaluator = COCOEvaluator("nucoco_val", ("bbox",), False, output_dir=args.output_dir)
    val_loader = build_detection_test_loader(cfg, "nucoco_val")
    results = inference_on_dataset(model, val_loader, evaluator)
    print("Evaluation results:", results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file path', required=True)
    parser.add_argument('--model-weights', dest='model_weights', help='Model weights path', required=True)
    parser.add_argument('--output-dir', dest='output_dir', help='Directory to save output images', required=True)
    parser.add_argument('--debug', action='store_true', help='If set, saves ground truth and inference images in debug folders')
    args = parser.parse_args()
    main(args)
