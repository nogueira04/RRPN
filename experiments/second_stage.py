import torch
import pickle
import cv2
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes, Instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.structures import BoxMode

# Load configuration and model
cfg = get_cfg()
cfg.merge_from_file("/clusterlivenfs/gnmp/RRPN/detectron2/configs/COCO-Detection/faster_rcnn_WideResNet_FPN.yaml")
# cfg.merge_from_file("/clusterlivenfs/gnmp/RRPN/configs/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
model = build_model(cfg)

model.eval()

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

# category_id_to_name = {0: "_", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "bus", 6: "truck"}
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

# Load the model weights
checkpointer = DetectionCheckpointer(model)
checkpointer.load("/clusterlivenfs/gnmp/RRPN/detectron2/output/wide_resnet_faster_rcnn/model_final.pth")
# checkpointer.load("/clusterlivenfs/gnmp/RRPN/data/models/faster_rcnn_X_101_32x8d_FPN_3x_nucoco_best_trial_2/model_final.pth")

# Load proposals from the proposal file
with open("/clusterlivenfs/gnmp/RRPN/data/nucoco/proposals/proposals_val.pkl", 'rb') as f:
    proposals = pickle.load(f)

# Map image ids to proposals for faster lookup
proposal_ids = set(proposals['ids'])
id_to_index = {img_id: idx for idx, img_id in enumerate(proposals['ids'])}

def load_image(image_path):
    # Load an image using OpenCV
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return img

def get_proposals(image_id):
    # Fetch proposals for the given image_id
    if image_id in proposal_ids:
        idx = id_to_index[image_id]
        return proposals['boxes'][idx], proposals['scores'][idx]
    else:
        raise ValueError(f"No proposals found for image_id {image_id}")

def draw_proposals(image, proposal_boxes):
    # Draw the proposals on the image (as rectangles)
    for box in proposal_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw in green with thickness 2
    return image

def save_images(original_image, proposals_image, original_path, proposals_path):
    # Save the original and proposals images
    cv2.imwrite(original_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))  # Convert to BGR for saving
    cv2.imwrite(proposals_path, cv2.cvtColor(proposals_image, cv2.COLOR_RGB2BGR))  # Convert to BGR for saving

def perform_inference(image_path, image_id):
    # Load the image
    image = load_image(image_path)
    
    # Get corresponding proposals
    proposal_boxes, proposal_scores = get_proposals(image_id)
    
    # Convert proposals to the required format for Detectron2
    instances = Instances(image.shape[:2])
    instances.proposal_boxes = Boxes(torch.tensor(proposal_boxes))
    instances.scores = torch.tensor(proposal_scores)
    
    # Convert image to tensor and send to model
    image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))  # HWC to CHW
    inputs = [{"image": image_tensor, "proposals": instances}]  # Note: "proposals" key added
    
    # Perform inference (run ROI heads with the proposals)
    with torch.no_grad():
        outputs = model(inputs)[0]
    
    return outputs, image, proposal_boxes

# Example usage:
image_id = 13874
image_path = "/clusterlivenfs/gnmp/RRPN/data/nucoco/val/00021647.jpg"

# Perform inference and get the original image and proposals
outputs, image, proposal_boxes = perform_inference(image_path, image_id)

# Save the original image
original_image_path = "/clusterlivenfs/gnmp/RRPN/output/original_image.jpg"
cv2.imwrite(original_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# Draw the proposals on the image
proposals_image = draw_proposals(image.copy(), proposal_boxes)

# Save the proposals image
proposals_image_path = "/clusterlivenfs/gnmp/RRPN/output/proposals_image.jpg"
cv2.imwrite(proposals_image_path, cv2.cvtColor(proposals_image, cv2.COLOR_RGB2BGR))

# Visualize the results
v = Visualizer(image, MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
v = Visualizer(image, MetadataCatalog.get("nucoco_val"), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Save the final image with predictions
output_image = out.get_image()[:, :, ::-1]  # Convert back to BGR for OpenCV
output_path = "/clusterlivenfs/gnmp/RRPN/output/predictions.jpg"  # Change to your desired save path
cv2.imwrite(output_path, output_image)

print(f"Images saved:\nOriginal: {original_image_path}\nProposals: {proposals_image_path}\nPredictions: {output_path}")
