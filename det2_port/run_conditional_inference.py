# File: scene_conditional_inference.py (Corrected - Loads Precomputed Map)

import argparse
import os
import sys
import time
import torch
import pickle
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np # Import numpy for proposal loading fallback
from tqdm import tqdm
import logging # Use standard logging

# --- Detectron2 Imports ---
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.structures import Boxes, Instances
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator

try:
    from nuscenes.nuscenes import NuScenes
except ImportError:
    print("WARNING: Failed to import NuScenes. nuscenes-devkit might be needed if other parts of the code use the NuScenes API.")
    NuScenes = None

# --- Local Timer Import ---
try:
    from timer import Timer
except ImportError:
    class Timer:
        def __init__(self):
            self.timings = {}
            self._start_times = {}

        def time(self, name):
            class TimerContext:
                def __init__(self, timer, name):
                    self.timer, self.name = timer, name

                def __enter__(self):
                    self.timer._start_times[self.name] = time.perf_counter()

                def __exit__(self, *a):
                    duration = time.perf_counter() - self.timer._start_times.pop(self.name, time.perf_counter())
                    self.timer.timings.setdefault(self.name, []).append(duration)

            return TimerContext(self, name)
    print("Warning: 'timer.py' not found. Using basic fallback Timer.")

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("scene_conditional_inference")

category_id_to_name = {0: "_", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "bus", 6: "truck"}

timer = Timer()

# --- Dataset Registration ---
def register_main_val_dataset(dataset_name, ann_file, img_dir):
    DatasetCatalog.register(dataset_name, lambda: load_coco_json(ann_file, img_dir, dataset_name=dataset_name))
    MetadataCatalog.get(dataset_name).set(thing_classes=list(category_id_to_name.values()))

    logger.info(f"Registered main validation dataset: {dataset_name} from {ann_file}")


# --- Proposal Loading ---
def load_proposals(proposal_file):
    logger.info(f"Loading proposals from: {proposal_file}")
    if not os.path.exists(proposal_file):
        raise FileNotFoundError(f"Proposal file not found: {proposal_file}")
    with timer.time(f"Load Proposals: {os.path.basename(proposal_file)}"):
        with open(proposal_file, 'rb') as f:
            proposals = pickle.load(f)
    if not all(key in proposals for key in ['ids', 'boxes', 'scores']):
         raise ValueError(f"Proposal file {proposal_file} missing 'ids', 'boxes', or 'scores'.")
    proposal_map = {
        int(img_id): (boxes, scores)
        for img_id, boxes, scores in zip(proposals['ids'], proposals['boxes'], proposals['scores'])
    }
    logger.info(f"Loaded {len(proposal_map)} proposals from {os.path.basename(proposal_file)}.")
    return proposal_map

def get_proposals_from_map(image_id, proposal_map):
    # Renamed np for clarity
    boxes_np, scores_np = proposal_map.get(image_id, (np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)))
    if boxes_np.size == 0:
        logger.warning(f"No proposals found for image_id {image_id}. Returning empty.")
    return boxes_np, scores_np


# --- Model Loading ---
def build_and_load_model(cfg_file, model_weights):
    """Helper to build and load a single model."""
    cfg = get_cfg()
    if not (cfg_file and os.path.exists(cfg_file)):
        raise ValueError(f"Config file {cfg_file} not found or invalid.")

    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.WEIGHTS = model_weights
    if "DATASETS" in cfg and "TEST" in cfg.DATASETS:
        cfg.DATASETS.TEST = ()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.freeze()

    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(model_weights)
    logger.info(f"Loaded model from weights: {model_weights} using config: {cfg_file}")
    return model, cfg


# --- Timing Graph Plotting ---
def plot_timing_graph(timing_summary, output_dir):
    steps = list(timing_summary.keys())
    if not steps:
        logger.warning("No timing data to plot.")
        return

    mean_times_ms = [(sum(times) / len(times) * 1000) if times else 0
                     for times in (timing_summary.get(step, []) for step in steps)]

    fig, ax = plt.subplots(figsize=(12, 7))
    width = 0.4
    x = np.arange(len(steps))
    bars = ax.bar(x, mean_times_ms, width, label="Mean Time (ms)", color='steelblue')

    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Pipeline Step")
    ax.set_title("Mean Execution Time per Step (Across All Processed Images)")
    ax.set_xticks(x)
    ax.set_xticklabels(steps, rotation=45, ha="right", fontsize=10)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va='bottom', fontsize=9)

    plt.tight_layout()
    graph_path = os.path.join(output_dir, "timing_graph.png")
    try:
        plt.savefig(graph_path)
        logger.info(f"Saved timing graph to {graph_path}")
    except Exception as e:
        logger.error(f"Failed to save timing graph: {e}")
    plt.close(fig)


# --- Main Inference Logic ---
def main(args):
    combined_dataset_name = "nucoco_val_combined"
    try:
        DatasetCatalog.clear()
        MetadataCatalog.clear()
        register_main_val_dataset(combined_dataset_name, args.val_ann_file, args.val_img_dir)
        metadata = MetadataCatalog.get(combined_dataset_name)
    except Exception as e:
        logger.error(f"Failed to register dataset or get metadata for {combined_dataset_name}: {e}", exc_info=True)
        sys.exit(1)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    logger.info(f"Loading image ID -> scene type map from: {args.scene_map_file}")
    if not os.path.exists(args.scene_map_file):
        logger.error(f"Scene mapping file not found: {args.scene_map_file}")
        sys.exit(1)
    try:
        with open(args.scene_map_file, 'rb') as f:
            image_id_to_scene_type = pickle.load(f)
        logger.info(f"Loaded scene map with {len(image_id_to_scene_type)} entries.")
        if not image_id_to_scene_type:
             logger.error("Loaded scene map is empty! Cannot proceed.")
             sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load scene mapping file: {e}", exc_info=True)
        sys.exit(1)
    # --- Mapping loaded ---

    # Register the main combined validation dataset


    # Load models
    logger.info("Loading models...")
    models, cfgs = {}, {}
    if not torch.cuda.is_available() and not args.allow_cpu:
        logger.error("CUDA is not available. Re-run with --allow-cpu only for an explicit CPU debug run.")
        sys.exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    try:
        with timer.time("Load Night Model"):
            models['night'], cfgs['night'] = build_and_load_model(args.cfg_night, args.weights_night)
        with timer.time("Load Rain Model"):
            models['rain'], cfgs['rain'] = build_and_load_model(args.cfg_rain, args.weights_rain)
        with timer.time("Load Other Model"):
            models['other'], cfgs['other'] = build_and_load_model(args.cfg_other, args.weights_other)
        for key in models:
            models[key].to(device)
    except Exception as e:
        logger.error(f"Failed to load one or more models: {e}", exc_info=True)
        sys.exit(1)

    # Load proposal sets
    logger.info("Loading proposals...")
    proposals = {}
    try:
        proposals['night'] = load_proposals(args.proposals_night)
        proposals['rain'] = load_proposals(args.proposals_rain)
        proposals['other'] = load_proposals(args.proposals_other)
    except Exception as e:
        logger.error(f"Failed to load one or more proposal files: {e}", exc_info=True)
        sys.exit(1)

    # Setup Evaluator
    evaluator = COCOEvaluator(combined_dataset_name, ("bbox",), False, output_dir=output_dir)
    evaluator.reset()

    # Custom Inference Loop
    try:
        dataset_dicts = DatasetCatalog.get(combined_dataset_name)
        if not dataset_dicts:
             logger.error(f"Dataset '{combined_dataset_name}' is empty. Check registration/JSON.")
             sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to get dataset dicts for {combined_dataset_name}: {e}")
        sys.exit(1)

    timing_summary = {}
    images_processed, images_skipped_mapping, images_skipped_proposals, images_skipped_load = 0, 0, 0, 0
    old_scene_type = None
    logger.info(f"Starting inference on {len(dataset_dicts)} images from {combined_dataset_name}...")
    for data_dict in tqdm(dataset_dicts, desc="Inference Progress"):
        timer.timings.clear()
        try:
            image_id = data_dict["image_id"]
            image_path = data_dict["file_name"]
            height = data_dict["height"]
            width = data_dict["width"]
        except KeyError as e:
            logger.warning(f"Data dict missing key {e}. Skipping.")
            continue

        # --- Determine scene type ---
        scene_type = image_id_to_scene_type.get(image_id) # Use loaded map
        if scene_type is None:
            logger.warning(f"Image ID {image_id} not found in loaded scene map. Skipping.")
            images_skipped_mapping += 1
            continue
        # --------------------------

        selected_model = models[scene_type]
        if scene_type != old_scene_type:
            logger.info(f"Switching model from {old_scene_type} to {scene_type} for image ID {image_id}.")
        old_scene_type = scene_type
        selected_proposals_map = proposals[scene_type]

        # Load Image
        if not os.path.exists(image_path):
            potential_path = os.path.join(metadata.image_root, image_path)
            if os.path.exists(potential_path):
                image_path = potential_path
            else:
                logger.warning(f"Image file not found: {image_path} or {potential_path}. Skipping {image_id}.")
                images_skipped_load += 1
                continue

        with timer.time("Loading image"):
            try:
                image = cv2.imread(image_path)
                if image is None:
                    raise IOError("cv2.imread returned None")
            except Exception as e:
                logger.warning(f"Failed to load image {image_path} ({image_id}): {e}")
                images_skipped_load += 1
                continue

        # Load Proposals
        with timer.time("Loading proposals"):
            try:
                proposal_boxes_np, proposal_scores_np = get_proposals_from_map(image_id, selected_proposals_map)
                proposal_boxes = torch.as_tensor(proposal_boxes_np, device=device)
                proposal_scores = torch.as_tensor(proposal_scores_np, device=device)
            except Exception as e:
                logger.warning(f"Error loading/processing proposals for {image_id}: {e}")
                images_skipped_proposals += 1
                continue

        # Prepare Proposals
        with timer.time("Prepare proposals"):
            instances = Instances((height, width))
            if proposal_boxes.numel() > 0:
                 instances.proposal_boxes = Boxes(proposal_boxes)
                 instances.scores = proposal_scores # Assuming model uses scores

        # Prepare Inputs
        with timer.time("Prepare inputs"):
            image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1), device=device)
            inputs = [{"image": image_tensor, "height": height, "width": width, "image_id": image_id, "proposals": instances}]

        # Perform Inference
        with timer.time("Perform inference"):
            with torch.no_grad():
                try:
                    outputs = selected_model(inputs)
                except Exception as e:
                    logger.error(f"Inference failed for {image_id} (scene: {scene_type}): {e}", exc_info=True)
                    continue

        # Process Results
        with timer.time("Process results"):
            try:
                evaluator.process(inputs, outputs)
            except Exception as e:
                logger.error(f"Evaluator processing failed for {image_id}: {e}", exc_info=True)

        # Collect Timing
        for step, times in timer.timings.items():
            timing_summary.setdefault(step, []).extend(times)
        images_processed += 1

        # Visualization / Debug
        if args.visualize or args.debug:
            with timer.time("Visualization"):
                try:
                    output_instances = outputs[0]["instances"].to("cpu")
                    v = Visualizer(image[:, :, ::-1], metadata, scale=1.2)
                    v_pred = v.draw_instance_predictions(output_instances)
                    result_img = v_pred.get_image()[:, :, ::-1]
                    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    vis_dir = os.path.join(output_dir, "visualizations")
                    os.makedirs(vis_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(vis_dir, f"pred_{scene_type}_{image_id}.jpg"), result_img)
                    if args.debug:
                        debug_dir = os.path.join(output_dir, 'debug', f"{image_id}_{scene_type}")
                        os.makedirs(debug_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(debug_dir, "inference.jpg"), result_img)
                        if "annotations" in data_dict:
                            v_gt = Visualizer(image[:, :, ::-1], metadata, scale=1.2).draw_dataset_dict(data_dict)
                            cv2.imwrite(os.path.join(debug_dir, "ground_truth.jpg"), v_gt.get_image()[:, :, ::-1])
                        else:
                            logger.warning(f"No GT annotations found for debug image {image_id}.")
                except Exception as e:
                    logger.warning(f"Visualization failed for {image_id}: {e}")

    # Final Evaluation
    logger.info(f"Inference loop complete. Processed: {images_processed}, Skipped (Mapping): {images_skipped_mapping}, Skipped (Proposals): {images_skipped_proposals}, Skipped (Load): {images_skipped_load}")
    if images_processed > 0:
        logger.info("Starting final evaluation...")
        try:
            results = evaluator.evaluate()
            logger.info(f"Evaluation results: {results}")
            results_path = os.path.join(output_dir, "evaluation_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            logger.info(f"Saved evaluation results to {results_path}")
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
        plot_timing_graph(timing_summary, output_dir)
    else:
        logger.warning("No images processed. Skipping evaluation/timing graph.")

    logger.info("Script finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test object detection models conditionally using precomputed scene map.')

    # --- Model Configs and Weights ---
    parser.add_argument('--cfg-night',  help='Config file path for the NIGHT model',
    default="/clusterlivenfs/gnmp/RRPN/data/models/faster_rcnn_X_152_32x8d_FPN_3x_finetune_night/faster_rcnn_X_152_32x8d_FPN_3x.yaml")
    parser.add_argument('--weights-night',  help='Model weights path for the NIGHT model',
    default="/clusterlivenfs/gnmp/RRPN/data/models/faster_rcnn_X_152_32x8d_FPN_3x_finetune_night/model_final.pth")
    parser.add_argument('--cfg-rain',  help='Config file path for the RAIN model',
    default="/clusterlivenfs/gnmp/RRPN/data/models/faster_rcnn_X_152_32x8d_FPN_3x_finetune_rain/faster_rcnn_X_152_32x8d_FPN_3x.yaml")
    parser.add_argument('--weights-rain',  help='Model weights path for the RAIN model',
    default="/clusterlivenfs/gnmp/RRPN/data/models/faster_rcnn_X_152_32x8d_FPN_3x_finetune_rain/model_final.pth")
    parser.add_argument('--cfg-other',  help='Config file path for the COMPLEMENT model',
    default="/clusterlivenfs/gnmp/RRPN/data/models/faster_rcnn_X_152_32x8d_FPN_3x_finetune_not_rain_and_night/faster_rcnn_X_152_32x8d_FPN_3x.yaml")
    parser.add_argument('--weights-other',  help='Model weights path for the COMPLEMENT model',
    default="/clusterlivenfs/gnmp/RRPN/data/models/faster_rcnn_X_152_32x8d_FPN_3x_finetune_not_rain_and_night/model_final.pth")

    # --- Proposal Files ---
    parser.add_argument('--proposals-night',  help='Proposal file path for the NIGHT validation split',
    default="/clusterlivenfs/gnmp/RRPN/data/nucoco/proposals/proposals_val_night.pkl")
    parser.add_argument('--proposals-rain',  help='Proposal file path for the RAIN validation split',
    default="/clusterlivenfs/gnmp/RRPN/data/nucoco/proposals/proposals_val_rain.pkl")
    parser.add_argument('--proposals-other',  help='Proposal file path for the COMPLEMENT validation split',
    default="/clusterlivenfs/gnmp/RRPN/data/nucoco/proposals/proposals_val_not_rain_and_night.pkl")

    parser.add_argument('--val-ann-file',  help='COCO JSON annotation file for the *entire* validation set',
    default="/clusterlivenfs/gnmp/RRPN/data/nucoco/annotations/instances_val.json")
    parser.add_argument('--val-img-dir',  help='Image directory for the *entire* validation set',
    default="/clusterlivenfs/gnmp/RRPN/data/nucoco/val")

    parser.add_argument('--scene-map-file', help='Path to the .pkl file mapping image_id -> scene_type.',
    default="/clusterlivenfs/gnmp/RRPN/data/nucoco/annotations/id_to_scene_val.pkl")

    # --- NuScenes Info ---
    parser.add_argument('--nusc-root',  help='Root directory of the NuScenes dataset',
    default="/clusterlivenfs/shared_datasets/nuscenes")
    parser.add_argument('--nusc-version', default='v1.0-trainval', help='NuScenes version (e.g., v1.0-trainval, v1.0-mini)')

    # --- Output and Options ---
    parser.add_argument('--output-dir',  help='Directory to save evaluation results and visualizations')
    parser.add_argument('--visualize', action='store_true', help='Save visualization images for predictions')
    parser.add_argument('--debug', action='store_true', help='Save detailed debug images (GT vs Pred) in subfolders')
    parser.add_argument('--allow-cpu', action='store_true', help='Allow CPU execution when CUDA is unavailable')
    parser.add_argument('-l', '--logging_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set logging level.')

    args = parser.parse_args()

    logger.setLevel(args.logging_level)

    main(args)
