import argparse
import os
import sys
import time
import torch
import pickle
import cv2
import yaml
import matplotlib.pyplot as plt
import numpy as np
from functools import partial  # Import partial
import torch.multiprocessing as mp # Import multiprocessing

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.structures import Boxes, Instances
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger # Added for better logging

# Imports needed for manual data loader construction
from detectron2.data import (
    DatasetFromList,
    DatasetMapper,
    build_batch_data_loader,
    MapDataset,
)
from detectron2.data.samplers import InferenceSampler

# Assuming timer.py exists and works as intended
# Create a dummy Timer class if timer.py is not available
try:
    from timer import Timer
except ImportError:
    print("Warning: 'timer.py' not found. Using dummy Timer.")
    class Timer:
        def __init__(self):
            self.timings = {}
            self._start_times = {}

        def time(self, name):
            return self._TimerContext(self, name)

        def print_summary(self):
            pass

        class _TimerContext:
            def __init__(self, timer_instance, name):
                self.timer = timer_instance
                self.name = name

            def __enter__(self):
                self.timer._start_times[self.name] = time.perf_counter()

            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.perf_counter()
                start_time = self.timer._start_times.pop(self.name, None)
                if start_time is not None:
                    duration = end_time - start_time
                    if self.name not in self.timer.timings:
                        self.timer.timings[self.name] = []
                    self.timer.timings[self.name].append(duration)


# Setup logger
setup_logger()

# --- Define root_dir calculation ---
current_path = os.path.dirname(os.path.abspath(__file__))
root_dir = None
temp_path = current_path
while True:
    if os.path.basename(temp_path) == 'RRPN':
        root_dir = temp_path
        break
    parent = os.path.dirname(temp_path)
    if parent == temp_path: # Reached root directory
        break
    temp_path = parent

if root_dir is None:
    print("Warning: Couldn't find 'RRPN' in path hierarchy. Ensure paths in configs/args are absolute or relative to execution dir.")
    root_dir = current_path # Example fallback

print(f"Identified root directory: {root_dir}")

# --- Load dataset config ---
dataset_config_path = "/clusterlivenfs/gnmp/RRPN/configs/general_config.yaml" # Adjust if needed
dataset_config = None
if os.path.exists(dataset_config_path):
    with open(dataset_config_path, "r") as f:
        dataset_config = yaml.safe_load(f)
        print(f"Loaded dataset config from: {dataset_config_path}")
else:
    relative_config_path = os.path.join(root_dir, "configs/general_config.yaml")
    print(f"Absolute path '{dataset_config_path}' not found. Trying relative path: '{relative_config_path}'")
    if os.path.exists(relative_config_path):
         with open(relative_config_path, "r") as f:
            dataset_config = yaml.safe_load(f)
            print(f"Loaded dataset config from relative path: {relative_config_path}")
    else:
        raise FileNotFoundError(f"Dataset config file not found at either '{dataset_config_path}' or '{relative_config_path}'")

# --- Extract paths from dataset config ---
try:
    val_config = dataset_config["NUCOCO"]["VAL"]
    train_config = dataset_config["NUCOCO"]["TRAIN"]
    val_img_dir = val_config["IMG_DIR"]
    val_ann_file = val_config["ANNOT_DIR"]
    train_img_dir = train_config["IMG_DIR"]
    train_ann_file = train_config["ANNOT_DIR"]
except KeyError as e:
    print(f"Error: Missing key in dataset config: {e}")
    sys.exit(1)

timer = Timer()

# --- Dataset registration ---
_DATASETS = {
    'nucoco_val': {'img_dir': val_img_dir, 'ann_file': val_ann_file},
    'nucoco_train': {'img_dir': train_img_dir, 'ann_file': train_ann_file},
}
category_id_to_name = {
    0: "bicycle", 1: "car", 2: "motorcycle", 3: "bus", 4: "person", 5: "truck",
}

def register_datasets():
    print("\nRegistering datasets...")
    for dataset_name, dataset_info in _DATASETS.items():
        print(f" Registering '{dataset_name}':")
        print(f"  Annotation file: {dataset_info['ann_file']}")
        print(f"  Image directory: {dataset_info['img_dir']}")
        if not os.path.exists(dataset_info['ann_file']): print(f"  WARNING: Annotation file not found!")
        if not os.path.exists(dataset_info['img_dir']): print(f"  WARNING: Image directory not found!")
        DatasetCatalog.register(dataset_name, lambda info=dataset_info: load_coco_json(info['ann_file'], info['img_dir'], dataset_name=dataset_name))
        MetadataCatalog.get(dataset_name).set(
            thing_classes=list(category_id_to_name.values()),
            evaluator_type="coco",
            json_file=dataset_info['ann_file'],
            image_root=dataset_info['img_dir']
        )
        print(f"  Registered '{dataset_name}' with classes: {list(category_id_to_name.values())}")

DatasetCatalog.clear()
MetadataCatalog.clear()
register_datasets()

# --- Proposal Loading Functions ---
def load_proposals(proposal_file):
    print(f"\nLoading proposals from: {proposal_file}")
    if not os.path.exists(proposal_file):
        raise FileNotFoundError(f"Proposal file not found: {proposal_file}")
    try:
        with open(proposal_file, 'rb') as f: proposals = pickle.load(f)
        if not isinstance(proposals, dict) or not all(k in proposals for k in ['ids', 'boxes', 'scores']):
            raise TypeError("Proposal file content is not a dict with 'ids', 'boxes', 'scores' keys.")
        if not (len(proposals['ids']) == len(proposals['boxes']) == len(proposals['scores'])):
             raise ValueError("Mismatch in lengths of 'ids', 'boxes', 'scores' in proposals.")
        print(f"Loaded {len(proposals.get('ids', []))} proposal entries.")
        return proposals
    except Exception as e: print(f"Error loading or parsing proposal file {proposal_file}: {e}"); raise

def get_proposals(image_id, proposals_dict, id_to_index_map):
    """Retrieves proposals for a given image ID using a precomputed index map."""
    if image_id in id_to_index_map:
        idx = id_to_index_map[image_id]
        proposal_boxes = proposals_dict['boxes'][idx]
        proposal_scores = proposals_dict['scores'][idx]
        if not isinstance(proposal_boxes, (list, torch.Tensor, np.ndarray)): raise TypeError(f"Proposal boxes for image {image_id} are not array-like.")
        if not isinstance(proposal_scores, (list, torch.Tensor, np.ndarray)): raise TypeError(f"Proposal scores for image {image_id} are not array-like.")

        proposal_boxes_np = np.asarray(proposal_boxes)
        proposal_scores_np = np.asarray(proposal_scores)

        num_proposals = proposal_boxes_np.shape[0] if proposal_boxes_np.ndim >= 1 else 0
        num_scores = proposal_scores_np.shape[0] if proposal_scores_np.ndim >= 1 else 0

        if num_proposals != num_scores:
             print(f"Warning: Mismatch between number of boxes ({num_proposals}) and scores ({num_scores}) for image {image_id}. Using minimum count.")
             min_count = min(num_proposals, num_scores)
             proposal_boxes_np = proposal_boxes_np[:min_count]
             proposal_scores_np = proposal_scores_np[:min_count]

        if proposal_boxes_np.ndim == 2 and proposal_boxes_np.shape[1] == 4 and proposal_scores_np.ndim == 1 and proposal_boxes_np.shape[0] == proposal_scores_np.shape[0]:
            return proposal_boxes_np, proposal_scores_np
        elif (proposal_boxes_np.ndim == 1 and proposal_boxes_np.shape[0] == 0) and (proposal_scores_np.ndim == 1 and proposal_scores_np.shape[0] == 0):
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        else:
            raise ValueError(f"Unexpected proposal shapes for image {image_id}. Boxes: {proposal_boxes_np.shape}, Scores: {proposal_scores_np.shape}")
    else:
        raise ValueError(f"No proposals found for image_id {image_id} in the loaded proposal file.")


# --- Helper function to add proposals during mapping (for evaluation loop) ---
def add_proposals_to_dict(data_dict, proposals_dict, proposal_id_to_index, device):
    """
    Takes a dict output by DatasetMapper, loads proposals using the
    globally available proposal data, and adds them to the dict.
    !!! IMPORTANT: This function will be called in DataLoader worker processes.
    !!! It should avoid initializing CUDA directly if possible, but moving
    !!! tensors to the target device is necessary here. 'spawn' start method is required.
    """
    image_id = data_dict.get("image_id")
    h = data_dict.get("height")
    w = data_dict.get("width")

    if image_id is None or h is None or w is None:
        print(f"ERROR: Missing 'image_id', 'height', or 'width' in data_dict for mapping. Keys: {data_dict.keys()}")
        instances = Instances((1, 1)) # Placeholder size
        instances.proposal_boxes = Boxes(torch.empty((0, 4), dtype=torch.float32, device=device))
        instances.scores = torch.empty((0,), dtype=torch.float32, device=device)
        data_dict["proposals"] = instances
        return data_dict

    try:
        proposal_boxes_np, proposal_scores_np = get_proposals(image_id, proposals_dict, proposal_id_to_index)
        proposal_boxes = torch.from_numpy(proposal_boxes_np).float()
        proposal_scores = torch.from_numpy(proposal_scores_np).float()

        instances = Instances((h, w))
        instances.proposal_boxes = Boxes(proposal_boxes)
        if proposal_scores.numel() > 0: instances.scores = proposal_scores
        else: instances.scores = torch.empty((0,), dtype=torch.float32) # Ensure scores field exists

        # Move instances to the target device *within the worker*
        # This might trigger the CUDA error if 'fork' is used. Requires 'spawn'.
        instances = instances.to(device)
        data_dict["proposals"] = instances

    except ValueError as e: # Handle case where get_proposals fails
        print(f"Warning: Could not load proposals for image {image_id} during mapping: {e}. Adding empty proposals.")
        instances = Instances((h, w))
        instances.proposal_boxes = Boxes(torch.empty((0, 4), dtype=torch.float32, device=device))
        instances.scores = torch.empty((0,), dtype=torch.float32, device=device)
        data_dict["proposals"] = instances
    except Exception as e:
        # Catch potential CUDA errors specifically if they happen here
        if "CUDA" in str(e):
             print(f"FATAL: Caught CUDA error in DataLoader worker for image {image_id}: {e}")
             print("Ensure multiprocessing start method is 'spawn'.")
             # Re-raise to stop the process, as recovery is unlikely
             raise e
        print(f"Unexpected error adding proposals for image {image_id}: {e}")
        print(f"ERROR: Adding empty proposals due to unexpected error for image {image_id}.")
        instances = Instances((h, w))
        instances.proposal_boxes = Boxes(torch.empty((0, 4), dtype=torch.float32, device=device))
        instances.scores = torch.empty((0,), dtype=torch.float32, device=device)
        data_dict["proposals"] = instances

    if "proposals" not in data_dict:
         print(f"CRITICAL WARNING: 'proposals' key still missing for {image_id} after add_proposals_to_dict. Adding empty.")
         instances = Instances((h, w))
         instances.proposal_boxes = Boxes(torch.empty((0, 4), dtype=torch.float32, device=device))
         instances.scores = torch.empty((0,), dtype=torch.float32, device=device)
         data_dict["proposals"] = instances

    return data_dict


# --- Utility Functions ---
def plot_timing_graph(timing_summary, output_dir):
    """Plots the mean execution time for each step."""
    valid_steps = {step: times for step, times in timing_summary.items() if times}
    if not valid_steps: print("No valid timing data to plot."); return
    steps = list(valid_steps.keys())
    mean_times = [ (sum(valid_steps[step]) / len(valid_steps[step])) * 1000 for step in steps]
    fig, ax = plt.subplots(figsize=(12, 6)); width = 0.4; x = np.arange(len(steps))
    bars1 = ax.bar(x, mean_times, width, label="Mean Time (ms)", color='skyblue', alpha=0.8)
    ax.set_ylabel("Time (ms)", fontsize=12); ax.set_xlabel("Pipeline Steps", fontsize=12)
    ax.set_title("Mean Execution Time per Step (Across All Processed Images)", fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels(steps, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=10); ax.grid(axis='y', linestyle='--', alpha=0.7)
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    autolabel(bars1); plt.tight_layout()
    graph_path = os.path.join(output_dir, "timing_graph_mean.png")
    try: plt.savefig(graph_path); print(f"Saved timing graph to {graph_path}")
    except Exception as e: print(f"Error saving timing graph: {e}")
    plt.close(fig)

def compute_iou(boxA, boxB):
    """Computes Intersection over Union (IoU) between two boxes."""
    if boxA[0] >= boxA[2] or boxA[1] >= boxA[3] or boxB[0] >= boxB[2] or boxB[1] >= boxB[3]: return 0.0
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6) # Add epsilon for stability
    return iou

# --- Inference function for the manual loop (visualization/debug) ---
def perform_inference(image_path, image_id, model, proposals_dict, id_to_index_map):
    """Performs inference on a single image using pre-loaded proposals."""
    with timer.time("Loading image"):
        image = cv2.imread(image_path)
        if image is None: print(f"Warning: Failed to load image at {image_path}. Skipping."); return None, None, None
        h, w = image.shape[:2]
    with timer.time("Loading proposals"):
        try:
            proposal_boxes_np, proposal_scores_np = get_proposals(image_id, proposals_dict, id_to_index_map)
            proposal_boxes = torch.from_numpy(proposal_boxes_np).float()
            proposal_scores = torch.from_numpy(proposal_scores_np).float()
        except (ValueError, IndexError, TypeError) as e: print(f"Error getting proposals for image {image_id}: {e}. Skipping inference."); return None, image, None
    with timer.time("Prepare proposals for Detectron2"):
        instances = Instances((h, w)); instances.proposal_boxes = Boxes(proposal_boxes)
        if proposal_scores.numel() > 0: instances.scores = proposal_scores
        else: print(f"Warning: No proposal scores found for image {image_id} in perform_inference. Scores field omitted.")
        try: instances = instances.to(model.device)
        except Exception as e: print(f"Error moving proposals to device {model.device}: {e}"); return None, image, None
    with timer.time("Prepare inputs"):
        image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image_tensor, "height": h, "width": w, "image_id": image_id, "proposals": instances}]
    with timer.time("Perform inference"):
        with torch.no_grad():
            try:
                outputs = model(inputs)
                if not outputs: print(f"Warning: Model returned empty output list for image {image_id}"); return None, image, proposal_boxes_np
                outputs = outputs[0]
            except Exception as e: print(f"Error during model inference for image {image_id}: {e}"); import traceback; traceback.print_exc(); return None, image, proposal_boxes_np
    if outputs and "instances" in outputs: outputs["instances"] = outputs["instances"].to("cpu")
    else: pass
    return outputs, image, proposal_boxes_np


# --- Main Execution ---
def main(args):
    # --- Load Configuration ---
    cfg = get_cfg()
    print(f"Loading configuration from: {args.cfg_file}")
    if args.cfg_file and os.path.exists(args.cfg_file):
        cfg.set_new_allowed(True); cfg.merge_from_file(args.cfg_file)
    else: raise ValueError(f"Config file {args.cfg_file} not found.")

    # --- Set Model Weights and Test Config ---
    cfg.MODEL.WEIGHTS = args.model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.conf_threshold
    cfg.DATASETS.TEST = ("nucoco_val",)
    proposal_file_path = args.proposal_file
    if not os.path.exists(proposal_file_path): raise FileNotFoundError(f"Specified proposal file not found: {proposal_file_path}")
    cfg.DATASETS.PROPOSAL_FILES_TEST = (proposal_file_path,)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {cfg.MODEL.DEVICE}")
    cfg.OUTPUT_DIR = args.output_dir
    cfg.freeze()
    print("Running with final configuration.")
    # print(cfg)

    # --- Build and Load Model ---
    print(f"\nBuilding model...")
    model = build_model(cfg)
    model.eval()
    target_device = model.device # Store the device the model is on
    print(f"Model device set to: {target_device}")
    print(f"Loading model weights from: {cfg.MODEL.WEIGHTS}")
    checkpointer = DetectionCheckpointer(model)
    load_result = checkpointer.load(cfg.MODEL.WEIGHTS)
    print("Model loaded successfully.")

    # --- Load Proposals (for viz/debug loop AND evaluation mapping) ---
    proposals_dict = load_proposals(proposal_file_path)
    proposal_id_to_index = {img_id: idx for idx, img_id in enumerate(proposals_dict['ids'])}
    print(f"Created index map for {len(proposal_id_to_index)} proposal IDs.")

    # --- Create Output Directory ---
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir); print(f"Created output directory: {args.output_dir}")

    # --- Get Full Dataset Dictionary List ---
    print("\nLoading full dataset metadata ('nucoco_val')...")
    try:
        full_dataset_dicts = DatasetCatalog.get("nucoco_val")
        if not full_dataset_dicts: print("Error: DatasetCatalog.get('nucoco_val') returned an empty list."); sys.exit(1)
        print(f"Loaded {len(full_dataset_dicts)} image entries for processing.")
    except KeyError: print("Error: Dataset 'nucoco_val' not found in DatasetCatalog. Check registration."); sys.exit(1)
    except Exception as e: print(f"Error loading dataset dicts: {e}"); sys.exit(1)
    metadata = MetadataCatalog.get("nucoco_val")

    timing_summary = {}; processed_image_count = 0

    # --- Main Inference Loop (Process all images for timing, viz, debug) ---
    print("\n--- Starting Inference Loop (Processing all images) ---")
    for i, d in enumerate(full_dataset_dicts):
        image_path = d.get("file_name")
        if not image_path or not os.path.exists(image_path): print(f"Warning: Skipping item {i} due to missing/invalid 'file_name': {image_path}"); continue
        image_id = d.get("image_id")
        if image_id is None:
            try: image_id = int(os.path.splitext(os.path.basename(image_path))[0])
            except (ValueError, IndexError): print(f"Warning: Could not parse image ID from filename: {os.path.basename(image_path)} and 'image_id' field missing. Skipping item {i}."); continue

        if (i + 1) % 100 == 0 or i == 0 or i == len(full_dataset_dicts) - 1: print(f"Processing image {i+1}/{len(full_dataset_dicts)}: {os.path.basename(image_path)} (ID: {image_id})")
        outputs, image, _ = perform_inference(image_path, image_id, model, proposals_dict, proposal_id_to_index)

        if outputs is not None and image is not None:
            processed_image_count += 1
            for step, times in timer.timings.items():
                if step not in timing_summary: timing_summary[step] = []
                timing_summary[step].extend(times)
        timer.timings.clear()

        # --- Visualization and Debug Output ---
        if image is not None and outputs is not None:
            v = Visualizer(image[:, :, ::-1], metadata, scale=1.2)
            pred_instances = outputs.get("instances")
            if pred_instances is not None and len(pred_instances) > 0:
                 out_vis = v.draw_instance_predictions(pred_instances)
                 result = out_vis.get_image()[:, :, ::-1]
            else: result = image # Use original if no predictions
            result_save_path = os.path.join(args.output_dir, f"prediction_{os.path.basename(image_path)}")
            cv2.imwrite(result_save_path, result)

            if args.debug:
                # Ground truth visualization
                v_gt = Visualizer(image[:, :, ::-1], metadata, scale=1.2); result_gt = None
                try:
                    if "annotations" in d and d["annotations"]:
                        out_gt = v_gt.draw_dataset_dict(d); result_gt = out_gt.get_image()[:, :, ::-1]
                    else: result_gt = image
                except Exception as e: print(f"Warning: Could not draw ground truth for {image_id}: {e}"); result_gt = image

                # FN/FP Calculation
                gt_boxes = []; fn_found = False; fp_found = False
                if "annotations" in d:
                    for ann in d["annotations"]:
                        if ann.get("iscrowd", 0) == 0 and "bbox" in ann:
                            x, y, w, h = ann["bbox"]
                            if w > 0 and h > 0: gt_boxes.append([x, y, x + w, y + h])
                            else: print(f"Warning: Skipping GT annotation with non-positive W/H for image {image_id}")

                pred_boxes = np.empty((0,4))
                if pred_instances is not None and pred_instances.has("pred_boxes") and pred_instances.has("scores"):
                    keep_idx = pred_instances.scores.cpu().numpy() >= args.conf_threshold
                    pred_boxes = pred_instances.pred_boxes.tensor.cpu().numpy()[keep_idx]

                iou_threshold = 0.5; gt_matched = [False] * len(gt_boxes); pred_matched = [False] * len(pred_boxes)
                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    for p_idx, p_box in enumerate(pred_boxes):
                        best_iou = -1; best_gt_idx = -1
                        for g_idx, g_box in enumerate(gt_boxes):
                            if not gt_matched[g_idx]:
                                iou = compute_iou(p_box, g_box)
                                if iou > best_iou: best_iou = iou; best_gt_idx = g_idx
                        if best_iou >= iou_threshold and best_gt_idx != -1:
                            if not gt_matched[best_gt_idx]: pred_matched[p_idx] = True; gt_matched[best_gt_idx] = True

                if not all(gt_matched): fn_found = True
                if not all(pred_matched): fp_found = True

                save_gt = result_gt if result_gt is not None else image; save_inf = result
                if fn_found:
                    debug_dir_fn = os.path.join(args.output_dir, 'debug_false_negative', str(image_id)); os.makedirs(debug_dir_fn, exist_ok=True)
                    cv2.imwrite(os.path.join(debug_dir_fn, f"{image_id}_ground_truth_FN.jpg"), save_gt)
                    cv2.imwrite(os.path.join(debug_dir_fn, f"{image_id}_inference_FN.jpg"), save_inf)
                    debug_dir_partial = os.path.join(args.output_dir, 'debug_not_all_found', str(image_id)); os.makedirs(debug_dir_partial, exist_ok=True)
                    cv2.imwrite(os.path.join(debug_dir_partial, f"{image_id}_ground_truth_partial.jpg"), save_gt)
                    cv2.imwrite(os.path.join(debug_dir_partial, f"{image_id}_inference_partial.jpg"), save_inf)
                if fp_found:
                    debug_dir_fp = os.path.join(args.output_dir, 'debug_false_positive', str(image_id)); os.makedirs(debug_dir_fp, exist_ok=True)
                    cv2.imwrite(os.path.join(debug_dir_fp, f"{image_id}_ground_truth_FP.jpg"), save_gt)
                    cv2.imwrite(os.path.join(debug_dir_fp, f"{image_id}_inference_FP.jpg"), save_inf)

    print(f"\n--- Finished processing loop. Processed {processed_image_count} images successfully. ---")

    # --- Plot Timing Graph ---
    if processed_image_count > 0 and timing_summary: print("Generating timing graph..."); plot_timing_graph(timing_summary, args.output_dir)
    else: print("Skipping timing graph (no images processed or no timing data).")

    # --- Prepare for Evaluation (on subset) ---
    num_to_skip = args.skip_eval_first_n; total_images = len(full_dataset_dicts)
    eval_dataset_dicts = []; eval_loader = None
    if num_to_skip < 0: print("Warning: skip_eval_first_n cannot be negative. Evaluating all images."); num_to_skip = 0

    if num_to_skip >= total_images:
        print(f"Skipping evaluation because number to skip ({num_to_skip}) >= total images ({total_images}).")
    else:
        start_eval_index = num_to_skip; eval_dataset_dicts = full_dataset_dicts[start_eval_index:]
        num_eval_images = len(eval_dataset_dicts)

        if num_eval_images > 0:
            print(f"\n--- Preparing Evaluation ---")
            print(f"Skipping first {num_to_skip} images for evaluation metrics.")
            print(f"Evaluating on the remaining {num_eval_images} images (indices {start_eval_index} to {total_images-1}).")
            print("Building evaluation data loader manually...")
            try:
                eval_dataset = DatasetFromList(eval_dataset_dicts, copy=False)
                base_mapper = DatasetMapper(cfg, is_train=False)
                # Create the mapper function that adds proposals
                final_mapper_callable = partial(add_proposals_to_dict,
                                                proposals_dict=proposals_dict,
                                                proposal_id_to_index=proposal_id_to_index,
                                                device=target_device) # Pass model's device
                # Define the combined mapping pipeline
                def final_mapper(dataset_dict):
                    data_dict = base_mapper(dataset_dict)
                    if data_dict is None:
                        # print(f"Warning: Base mapper returned None for {dataset_dict.get('file_name')}, skipping.")
                        return None # Important to return None if base mapping fails
                    data_dict = final_mapper_callable(data_dict)
                    return data_dict

                mapped_eval_dataset = MapDataset(eval_dataset, final_mapper)
                sampler = InferenceSampler(num_eval_images)
                images_per_batch = cfg.TEST.get("IMS_PER_BATCH", cfg.SOLVER.get("IMS_PER_BATCH", 1))
                if images_per_batch <= 0: print(f"Warning: Invalid IMS_PER_BATCH ({images_per_batch}), defaulting to 1."); images_per_batch = 1
                print(f"Using evaluation batch size: {images_per_batch}")
                # Determine num_workers from config, default to 0 if not specified
                num_workers = cfg.DATALOADER.get("NUM_WORKERS", 0)
                print(f"Using num_workers: {num_workers}")
                eval_loader = build_batch_data_loader(
                    dataset=mapped_eval_dataset, sampler=sampler,
                    total_batch_size=images_per_batch, aspect_ratio_grouping=False,
                    num_workers=num_workers # Use value from config or 0
                )
                print("Successfully built evaluation data loader for the subset.")
            except Exception as e:
                 print(f"\nError: Failed to manually build evaluation data loader: {e}"); import traceback; traceback.print_exc(); eval_loader = None

            # --- Run Evaluation ---
            if eval_loader:
                evaluator_name = cfg.DATASETS.TEST[0]; output_eval_dir = os.path.join(args.output_dir, "coco_eval")
                os.makedirs(output_eval_dir, exist_ok=True)
                print(f"Initializing COCOEvaluator for '{evaluator_name}'...")
                evaluator = COCOEvaluator(evaluator_name, ("bbox",), False, output_dir=output_eval_dir)
                evaluator.reset()
                print(f"\nRunning inference for evaluation on the {num_eval_images} image subset...")
                try:
                    results = inference_on_dataset(model, eval_loader, evaluator)
                    print("\n--- Evaluation Results (Subset) ---")
                    if results: print(results); print(f"Evaluation results saved in: {output_eval_dir}")
                    else: print("Evaluation did not produce results (or evaluator returned None).")
                except AssertionError as ae:
                    print(f"\nCaught AssertionError during inference_on_dataset: {ae}")
                    print("This likely means the 'proposals' key was still missing in some input batches.")
                    print("Please check the 'add_proposals_to_dict' function and mapping pipeline carefully.")
                    import traceback; traceback.print_exc()
                except Exception as e_inf:
                    print(f"\nAn error occurred during evaluation inference: {e_inf}")
                    import traceback; traceback.print_exc()

            else: print("Skipping evaluation due to data loader build failure.")
        else: print("No images remaining for evaluation after skipping.")


if __name__ == "__main__":
    # === Set Multiprocessing Start Method (IMPORTANT for CUDA) ===
    # This should be done *before* any CUDA tensors or operations are created
    # in worker processes, which can happen during DataLoader construction or mapping.
    try:
        # Check if CUDA is available BEFORE setting the method, as it might not be needed if only CPU is used.
        if torch.cuda.is_available():
             mp.set_start_method('spawn', force=True)
             print("Multiprocessing start method set to 'spawn' for CUDA compatibility.")
        else:
             print("CUDA not available, default multiprocessing start method will be used.")
    except RuntimeError as e:
        # Check if the context is already set correctly
        current_context = mp.get_start_method(allow_none=True)
        if current_context != 'spawn' and torch.cuda.is_available():
            print(f"Warning: Could not set start method to 'spawn': {e}")
            print(f"Current start method: {current_context}")
            print("CUDA errors might occur in DataLoader workers if NUM_WORKERS > 0.")
        elif current_context == 'spawn':
             print("Multiprocessing start method already set to 'spawn'.")
        # else: context is something else, but CUDA not available, so less critical.

    # === Argument Parsing ===
    parser = argparse.ArgumentParser(description='Test a detection model with pre-computed proposals (for viz/debug) and evaluate on a subset.')
    parser.add_argument('--cfg', dest='cfg_file', help='Detectron2 config file path (.yaml)', required=True)
    parser.add_argument('--model-weights', dest='model_weights', help='Model weights path (.pth)', required=True)
    parser.add_argument('--proposal-file', help='Path to pre-computed proposals (.pkl) used for visualization/debug loop AND evaluation loop', required=True)
    parser.add_argument('--output-dir', dest='output_dir', help='Directory to save output images, timing, and evaluation results', required=True)
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='Confidence threshold for filtering detections during visualization/debug analysis (NOTE: evaluation uses threshold from config/model)')
    parser.add_argument('--debug', action='store_true', help='Save ground truth and inference images in debug folders, especially highlighting FN/FP cases.')
    parser.add_argument('--skip-eval-first-n', type=int, default=0, help='Skip the first N images when calculating final COCO evaluation metrics. Default is 0.')
    args = parser.parse_args()

    # === Basic Input Validation ===
    if not os.path.exists(args.cfg_file): print(f"Error: Config file not found at {args.cfg_file}"); sys.exit(1)
    if not os.path.exists(args.model_weights): print(f"Error: Model weights file not found at {args.model_weights}"); sys.exit(1)
    if not os.path.exists(args.proposal_file): print(f"Error: Proposal file not found at {args.proposal_file}"); sys.exit(1)
    if args.skip_eval_first_n < 0: print("Warning: --skip-eval-first-n cannot be negative. Setting to 0."); args.skip_eval_first_n = 0

    # === Run Main Function ===
    try:
        main(args)
    except Exception as e:
        print(f"\nAn critical error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        pass # Optional cleanup

    print("\nScript finished.")