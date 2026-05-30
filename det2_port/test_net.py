import argparse
import os
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

timer = Timer()

category_id_to_name = {
    0: "bicycle",
    1: "car",
    2: "motorcycle",
    3: "bus",
    4: "person",
    5: "truck",
}

def find_repo_root(start_path):
    current = start_path
    while os.path.basename(current) != "RRPN":
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError("Couldn't find 'RRPN' in path hierarchy.")
        current = parent
    return current

def load_dataset_paths(dataset_config_path):
    with open(dataset_config_path, "r") as f:
        dataset_config = yaml.safe_load(f)
    val_config = dataset_config["NUCOCO"]["VAL"]
    return val_config["ANNOT_DIR"], val_config["IMG_DIR"]

def register_dataset(dataset_name, ann_file, img_dir):
    DatasetCatalog.clear()
    MetadataCatalog.clear()
    DatasetCatalog.register(
        dataset_name,
        lambda: load_coco_json(ann_file, img_dir, dataset_name=dataset_name),
    )
    MetadataCatalog.get(dataset_name).set(thing_classes=list(category_id_to_name.values()))

# --- MODIFIED FUNCTION TO DRAW WITH CONFIDENCE, THICKER BOXES, AND LARGER TEXT ---
def draw_custom_style_predictions(image, outputs, metadata):
    """
    Draws bounding boxes and labels on an image in a custom style, including
    confidence scores, with thicker lines and larger text.

    Args:
        image (np.ndarray): The image to draw on (in BGR format).
        outputs (dict): The output from the Detectron2 model.
        metadata: The metadata for the dataset, used to get class names.

    Returns:
        np.ndarray: The image with visualizations.
    """
    # Define colors for each class name (BGR format)
    COLOR_MAP = {
        'car': (64, 63, 255),        # Red
        'person': (53, 183, 253),   # Orange
        'bicycle': (255, 0, 0),    # Blue
        'motorcycle': (255, 0, 255),# Magenta
        'bus': (0, 128, 255),      # A different shade of Orange
        'truck': (0, 69, 255),     # Brownish-orange
    }
    DEFAULT_COLOR = (128, 128, 128) # Gray for other classes

    img_with_boxes = image.copy()
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()
    class_names = metadata.get("thing_classes", [])

    # --- UPDATED DRAWING PROPERTIES ---
    box_thickness = 4          # Increased from 2 to 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7           # Increased from 0.5 to 0.7 for bigger text
    font_thickness = 2         # Increased for better text visibility
    text_color = (255, 255, 255) # White
    connector_color = (255, 255, 255) # White
    connector_thickness = 2
    label_padding = 5
    connector_height = 10

    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        class_id = classes[i]
        score = scores[i]
        class_name = class_names[class_id] if class_id < len(class_names) else "N/A"
        box_color = COLOR_MAP.get(class_name, DEFAULT_COLOR)

        # 1. Draw the main bounding box (now thicker)
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), box_color, box_thickness)

        # 2. Prepare the label text with class name and confidence score
        label_text = f"{class_name}: {score:.0%}" # e.g., "car: 99%"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        # 3. Define and draw the label's background rectangle
        label_tl_y = y1 - connector_height - text_h - (2 * label_padding)
        # Handle cases where the box is at the top of the image
        if label_tl_y < 0:
            label_tl_y = y1 + connector_height + box_thickness

        label_tl_x = x1
        label_br_x = x1 + text_w + (2 * label_padding)
        label_br_y = label_tl_y + text_h + (2 * label_padding)

        cv2.rectangle(img_with_boxes, (label_tl_x, label_tl_y), (label_br_x, label_br_y), box_color, -1)

        # 4. Put the class name and score text on the label background (now larger)
        text_pos = (x1 + label_padding, label_tl_y + text_h + label_padding)
        cv2.putText(img_with_boxes, label_text, text_pos, font, font_scale, text_color, font_thickness)

        # 5. Draw the connector line
        connector_start = (x1, y1)
        # Adjust connector end based on label position
        if label_tl_y < y1:
            connector_end = (x1, y1 - connector_height)
            cv2.line(img_with_boxes, connector_start, connector_end, connector_color, connector_thickness)

    return img_with_boxes

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
    mean_times = [sum(timing_summary[step]) / len(timing_summary[step]) for step in steps] if steps else []

    if not mean_times:
        print("No timing data to plot.")
        return

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
        instances = Instances(image.shape[:2])
        instances.proposal_boxes = Boxes(torch.tensor(proposal_boxes))
        instances.scores = torch.tensor(proposal_scores)

    with timer.time("Prepare inputs"):
        image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image_tensor, "proposals": instances}]

    with timer.time("Perform inference"):
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
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

def main(args):
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation. Remove --require-cuda only for an explicit CPU debug run.")

    repo_root = find_repo_root(current_path)
    dataset_config_path = args.dataset_config or os.path.join(repo_root, "configs/general_config.yaml")
    ann_file = args.ann_file
    img_dir = args.img_dir
    if not ann_file or not img_dir:
        ann_file, img_dir = load_dataset_paths(dataset_config_path)
    register_dataset(args.dataset_name, ann_file, img_dir)

    cfg = get_cfg()
    if args.cfg_file and os.path.exists(args.cfg_file):
        cfg.set_new_allowed(True)
        cfg.merge_from_file(args.cfg_file)
    else:
        raise ValueError(f"Config file {args.cfg_file} not found.")

    cfg.MODEL.WEIGHTS = args.model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TEST = (args.dataset_name,)

    proposal_file = args.proposal_file
    if not os.path.exists(proposal_file):
        raise FileNotFoundError(f"Proposal file not found: {proposal_file}")
    cfg.DATASETS.PROPOSAL_FILES_TEST = (proposal_file,)

    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(args.model_weights)
    print(cfg)
    proposals = load_proposals(proposal_file)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    timing_summary = {}
    dataset_dicts = DatasetCatalog.get(args.dataset_name)

    for d in dataset_dicts:
        image_path = d["file_name"]
        if not os.path.exists(image_path):
            print(f"Skipping missing image: {image_path}")
            continue

        image_id = d.get("image_id", int(os.path.splitext(os.path.basename(image_path))[0]))

        try:
            outputs, image, proposal_boxes = perform_inference(image_path, image_id, model, proposals)
        except ValueError as e:
            print(f"Skipping image_id {image_id} due to error: {e}")
            continue

        for step in timer.timings:
            if step not in timing_summary:
                timing_summary[step] = []
            timing_summary[step].extend(timer.timings[step])

        timer.print_summary()
        timer.timings.clear()

        # --- MODIFICATION: USE THE NEW DRAWING FUNCTION ---
        # Instead of using Visualizer, call our custom function
        result = draw_custom_style_predictions(image, outputs, MetadataCatalog.get(args.dataset_name))
        result_save_path = os.path.join(args.output_dir, f"predictions_{os.path.basename(image_path)}")
        cv2.imwrite(result_save_path, result)
        print(f"Saved prediction to {result_save_path}")

        if args.debug:
            debug_dir = os.path.join(args.output_dir, 'debug', f"{image_id}")
            os.makedirs(debug_dir, exist_ok=True)

            # Use default Visualizer for ground truth for simplicity
            v_gt = Visualizer(image[:, :, ::-1], MetadataCatalog.get(args.dataset_name), scale=1.2)
            v_gt = v_gt.draw_dataset_dict(d)
            result_gt = v_gt.get_image()[:, :, ::-1]
            cv2.imwrite(os.path.join(debug_dir, "ground_truth.jpg"), result_gt)

            # Save the custom-styled inference image
            cv2.imwrite(os.path.join(debug_dir, "inference.jpg"), result)

            gt_boxes = [ann["bbox"] for ann in d.get("annotations", [])]
            # Convert [x,y,w,h] to [x1,y1,x2,y2]
            gt_boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in gt_boxes]

            pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

            iou_threshold = 0.5
            fn_found = False
            fp_found = False

            for g in gt_boxes:
                if not pred_boxes.size or max([compute_iou(g, p) for p in pred_boxes]) < iou_threshold:
                    fn_found = True
                    break

            for p in pred_boxes:
                if not gt_boxes or max([compute_iou(p, g) for g in gt_boxes]) < iou_threshold:
                    fp_found = True
                    break

            if fn_found:
                debug_dir_fn = os.path.join(args.output_dir, 'debug_false_negative', str(image_id))
                os.makedirs(debug_dir_fn, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir_fn, "ground_truth.jpg"), result_gt)
                cv2.imwrite(os.path.join(debug_dir_fn, "inference.jpg"), result)

            if fp_found:
                debug_dir_fp = os.path.join(args.output_dir, 'debug_false_positive', str(image_id))
                os.makedirs(debug_dir_fp, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir_fp, "ground_truth.jpg"), result_gt)
                cv2.imwrite(os.path.join(debug_dir_fp, "inference.jpg"), result)

    plot_timing_graph(timing_summary, args.output_dir)

    # Evaluation part
    try:
        evaluator = COCOEvaluator(args.dataset_name, ("bbox",), False, output_dir=args.output_dir)
        val_loader = build_detection_test_loader(cfg, args.dataset_name)
        results = inference_on_dataset(model, val_loader, evaluator)
        print("Evaluation results:", results)
    except Exception as e:
        print(f"Could not run COCO evaluation. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network with custom visualization')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file path', required=True)
    parser.add_argument('--model-weights', dest='model_weights', help='Model weights path', required=True)
    parser.add_argument('--output-dir', dest='output_dir', help='Directory to save output images', required=True)
    parser.add_argument('--proposal-file', required=True, help='RRPN proposal pickle for the evaluated split')
    parser.add_argument('--ann-file', help='COCO annotation JSON for the evaluated split')
    parser.add_argument('--img-dir', help='Image directory for the evaluated split')
    parser.add_argument('--dataset-name', default='nucoco_val', help='Detectron2 dataset name to register for evaluation')
    parser.add_argument('--dataset-config', help='Fallback YAML with NUCOCO.VAL paths when --ann-file/--img-dir are omitted')
    parser.add_argument('--require-cuda', action='store_true', help='Fail instead of silently evaluating on CPU')
    parser.add_argument('--debug', action='store_true', help='If set, saves ground truth and inference images in debug folders')
    args = parser.parse_args()
    main(args)
