import torch
import cv2
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Load configuration and model
cfg = get_cfg()
cfg.merge_from_file("/clusterlivenfs/gnmp/RRPN/configs/fast_rcnn_R_50_FPN_1x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a threshold for inference
model = build_model(cfg)
model.eval()

# Load the model weights
checkpointer = DetectionCheckpointer(model)
checkpointer.load("/clusterlivenfs/gnmp/RRPN/data/models/model_final_e5f7ce.pkl")

def load_image(image_path):
    # Load an image using OpenCV
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return img

def perform_inference(image_path):
    # Load the image
    image = load_image(image_path)
    
    # Convert image to tensor and prepare for model input
    image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))  # HWC to CHW
    inputs = [{"image": image_tensor}]
    
    # Perform inference directly on the image
    with torch.no_grad():
        # This assumes the model generates its own proposals
        outputs = model.inference(inputs)
    
    return outputs[0], image

# Example usage:
image_path = "/clusterlivenfs/gnmp/RRPN/data/nucoco/val/00004598.jpg"

# Perform inference directly on the image
outputs, image = perform_inference(image_path)

# Visualize the results
v = Visualizer(image, MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Save the final image with predictions
output_image = out.get_image()[:, :, ::-1]  # Convert back to BGR for OpenCV
output_path = "/clusterlivenfs/gnmp/RRPN/output/predictions_no_proposals.jpg"  # Change to your desired save path
cv2.imwrite(output_path, output_image)

print(f"Prediction image saved at: {output_path}")
