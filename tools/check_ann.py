import os
import json
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import logging
import random

# --- Optional: Supervision for nicer drawing ---
try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    print("INFO: 'supervision' library not found. Using basic OpenCV drawing.")
    print("      Install with: pip install supervision")
# ---------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize COCO annotations on images.')
    parser.add_argument('--ann_file', required=True,
                        help='Path to the COCO annotation JSON file (e.g., output_ann/annotations/instances_val.json).')
    parser.add_argument('--img_dir', required=True,
                        help='Path to the directory containing the images (e.g., output_ann/val/). This directory should contain the subfolders like samples/CAM_FRONT.')
    parser.add_argument('--output_dir', required=True,
                        help='Path to the directory where annotated images will be saved.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Optional: Limit the number of images to process for testing.')
    parser.add_argument('--use_supervision', action='store_true',
                        help='Use supervision library for drawing (recommended if installed).')
    parser.add_argument('--show_score', action='store_true',
                        help='Display confidence score on the bounding box label.')
    parser.add_argument('--color', type=int, nargs=3, default=None, metavar=('B', 'G', 'R'),
                        help='Specify BGR color for boxes (e.g., --color 0 255 0 for green). Random if not specified.')
    parser.add_argument('--thickness', type=int, default=2,
                        help='Thickness of the bounding box lines.')

    args = parser.parse_args()
    if args.use_supervision and not SUPERVISION_AVAILABLE:
        logger.warning("Supervision requested but not available. Falling back to OpenCV drawing.")
        args.use_supervision = False
    return args

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

def draw_opencv(image, annotations, cat_id_to_info, color_map, thickness, show_score):
    """Draws annotations using basic OpenCV functions."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]
    for ann in annotations:
        cat_id = ann['category_id']
        cat_info = cat_id_to_info.get(cat_id)
        if not cat_info: continue # Skip if category not found

        cat_name = cat_info['name']
        color = color_map.get(cat_id)
        if color is None: # Assign random color if not seen before
            color = tuple(np.random.randint(0, 256, size=3).tolist())
            color_map[cat_id] = color

        x, y, w, h = map(int, ann['bbox']) # COCO format [xmin, ymin, width, height]

        # Ensure coordinates are within image bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w - 1, x + w)
        y2 = min(img_h - 1, y + h)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Prepare label text
        label = cat_name
        if show_score and 'score' in ann:
            label += f" {ann['score']:.2f}"

        # Calculate text size and position
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1, label_height + 5) # Position below top, but ensure it's within image
        label_x = x1
        label_bg_y2 = label_y + 5
        label_bg_x2 = label_x + label_width

        # Draw filled rectangle as background for text
        cv2.rectangle(image, (label_x - 2, label_y - label_height - 5), (label_bg_x2 + 2, label_bg_y2 -5 ), color, -1)
        # Put text
        cv2.putText(image, label, (label_x, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # White text

    return image


def draw_supervision(image, annotations, cat_id_to_info, show_score, font_scale=1.15, font_thickness=2):
    """
    Draws annotations using the supervision library with adjustable font size and thickness.
    """
    # Convert COCO annotations to supervision Detections format
    boxes = []
    class_ids = []
    confidences = []
    labels = []

    for ann in annotations:
        x, y, w, h = ann['bbox']
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        x1, y1, x2, y2 = x, y, x + w, y + h
        boxes.append([x1, y1, x2, y2])

        cat_id = ann['category_id']
        class_ids.append(cat_id)

        score = ann.get('score', None) # Get score if available
        confidences.append(score if score is not None else 1.0) # Default confidence if none

        cat_name = cat_id_to_info.get(cat_id, {}).get('name', 'UNK')
        label = cat_name
        if show_score and score is not None:
            label += f" {score:.2f}"
        labels.append(label)

    if not boxes:
        return image # Return original image if no boxes to draw

    detections = sv.Detections(
        xyxy=np.array(boxes),
        class_id=np.array(class_ids),
        confidence=np.array(confidences)
    )

    # Use supervision annotators
    box_annotator = sv.BoxAnnotator()
    # Increase the font size and thickness by setting the parameters
    label_annotator = sv.LabelAnnotator(
        text_scale=font_scale,
        text_thickness=font_thickness
    )

    annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    return annotated_image

def main():
    args = parse_args()

    # --- Load COCO JSON ---
    logger.info(f"Loading annotations from: {args.ann_file}")
    if not os.path.exists(args.ann_file):
        logger.error(f"Annotation file not found: {args.ann_file}")
        sys.exit(1)
    with open(args.ann_file, 'r') as f:
        coco_data = json.load(f)
    logger.info("Annotations loaded.")

    # --- Prepare data structures ---
    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])

    if not images:
        logger.error("No 'images' found in the annotation file.")
        sys.exit(1)

    # Create mapping from category ID to category info (name, etc.)
    cat_id_to_info = {cat['id']: cat for cat in categories}
    logger.info(f"Found {len(cat_id_to_info)} categories.")

    # Create mapping from image ID to image info (for quick lookup)
    img_id_to_info = {img['id']: img for img in images}

    # Create mapping from image ID to list of annotations
    img_id_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    logger.info(f"Processed {len(annotations)} annotations for {len(img_id_to_anns)} images.")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Saving annotated images to: {args.output_dir}")

    # --- Prepare drawing specifics ---
    color_map = {} # For consistent random colors in OpenCV mode
    if args.color: # Use specified color if provided
         # Convert single specified color for all boxes
         fixed_color = tuple(args.color)
         color_map = {cat_id: fixed_color for cat_id in cat_id_to_info}


    # --- Process Images ---
    processed_count = 0
    image_list = images[:args.limit] if args.limit else images

    for img_info in tqdm(image_list, desc="Annotating Images"):
        img_id = img_info['id']
        relative_img_path = img_info['file_name']
        full_img_path = os.path.join(args.img_dir, relative_img_path)

        if not os.path.exists(full_img_path):
            logger.warning(f"Image file not found, skipping: {full_img_path}")
            continue

        # Load image
        try:
            image = cv2.imread(full_img_path)
            if image is None:
                logger.warning(f"Failed to load image, skipping: {full_img_path}")
                continue
        except Exception as e:
            logger.error(f"Error loading image {full_img_path}: {e}")
            continue

        # Get annotations for this image
        current_anns = img_id_to_anns.get(img_id, [])

        # Draw annotations
        if current_anns: # Only draw if there are annotations
             if args.use_supervision:
                  annotated_image = draw_supervision(image, current_anns, cat_id_to_info, args.show_score)
             else:
                  annotated_image = draw_opencv(image, current_anns, cat_id_to_info, color_map, args.thickness, args.show_score)
        else:
             annotated_image = image # Keep original if no annotations

        # Prepare output path, mirroring subdirectory structure
        output_path = os.path.join(args.output_dir, relative_img_path)
        output_subdir = os.path.dirname(output_path)
        os.makedirs(output_subdir, exist_ok=True)

        # Save annotated image
        try:
            cv2.imwrite(output_path, annotated_image)
            processed_count += 1
        except Exception as e:
            logger.error(f"Failed to save annotated image {output_path}: {e}")

    logger.info(f"Finished. Processed and saved {processed_count} annotated images.")


if __name__ == "__main__":
    main()