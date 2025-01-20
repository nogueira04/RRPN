import argparse
import os
import yaml
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
import detectron2.modeling.backbone.wide_resnet

category_id_to_name = {0: "_", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "bus", 6: "truck"}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference script with Detectron2.")
    parser.add_argument("--config-file", required=True, help="Path to the model configuration file.")
    parser.add_argument("--weights-file", required=True, help="Path to the model weights file.")
    parser.add_argument("--image-path", required=True, help="Path to the image file for inference.")
    parser.add_argument("--proposals-file", required=True, help="Path to the proposals file (pickle format).")
    parser.add_argument("--output-dir", required=True, help="Directory to save the output images.")
    parser.add_argument("--debug", action="store_true", help="Save the original and proposals images if enabled.")
    return parser.parse_args()

def load_dataset_config(dataset_config_path):
    with open(dataset_config_path, "r") as f:
        return yaml.safe_load(f)

def register_datasets(dataset_config):
    val_config = dataset_config["NUCOCO"]["VAL"]
    train_config = dataset_config["NUCOCO"]["TRAIN"]

    datasets = {
        'nucoco_val': {
            'img_dir': val_config["IMG_DIR"],
            'ann_file': val_config["ANNOT_DIR"],
        },
        'nucoco_train': {
            'img_dir': train_config["IMG_DIR"],
            'ann_file': train_config["ANNOT_DIR"],
        },
    }

    for dataset_name, dataset_info in datasets.items():
        DatasetCatalog.register(dataset_name, lambda dataset_info=dataset_info: load_coco_json(dataset_info['ann_file'], dataset_info['img_dir']))
        MetadataCatalog.get(dataset_name).set(thing_classes=list(category_id_to_name.values()))

def load_image(image_path):
    img = cv2.imread(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def get_image_id(image_path):
    filename = os.path.basename(image_path)
    image_id = int(os.path.splitext(filename)[0])
    return image_id

def get_proposals(image_id, proposals, proposal_ids, id_to_index):
    if image_id in proposal_ids:
        idx = id_to_index[image_id]
        return proposals['boxes'][idx], proposals['scores'][idx]
    else:
        raise ValueError(f"No proposals found for image_id {image_id}")

def draw_proposals(image, proposal_boxes):
    for box in proposal_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

def perform_inference(image_path, image_id, proposals, proposal_ids, id_to_index, model):
    image = load_image(image_path)
    proposal_boxes, proposal_scores = get_proposals(image_id, proposals, proposal_ids, id_to_index)

    instances = Instances(image.shape[:2])
    instances.proposal_boxes = Boxes(torch.tensor(proposal_boxes))
    instances.scores = torch.tensor(proposal_scores)

    image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image_tensor, "proposals": instances}]

    with torch.no_grad():
        outputs = model(inputs)[0]

    return outputs, image, proposal_boxes

def main():
    args = parse_arguments()

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    model = build_model(cfg)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(args.weights_file)

    with open(args.proposals_file, 'rb') as f:
        proposals = pickle.load(f)

    proposal_ids = set(proposals['ids'])
    id_to_index = {img_id: idx for idx, img_id in enumerate(proposals['ids'])}

    root_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_config_path = os.path.join(root_dir, "configs/general_config.yaml")
    dataset_config = load_dataset_config("/clusterlivenfs/gnmp/RRPN/configs/general_config.yaml")
    register_datasets(dataset_config)

    image_id = get_image_id(args.image_path)
    outputs, image, proposal_boxes = perform_inference(
        args.image_path, image_id, proposals, proposal_ids, id_to_index, model
    )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.debug:
        original_image_path = os.path.join(args.output_dir, "original_image.jpg")
        cv2.imwrite(original_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Original image saved at: {original_image_path}")

        proposals_image = draw_proposals(image.copy(), proposal_boxes)
        proposals_image_path = os.path.join(args.output_dir, "proposals_image.jpg")
        cv2.imwrite(proposals_image_path, cv2.cvtColor(proposals_image, cv2.COLOR_RGB2BGR))
        print(f"Proposals image saved at: {proposals_image_path}")

    v = Visualizer(image, MetadataCatalog.get("nucoco_val"), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    output_image = out.get_image()[:, :, ::-1]
    predictions_path = os.path.join(args.output_dir, "predictions.jpg")
    cv2.imwrite(predictions_path, output_image)

    print(f"Predictions image saved at: {predictions_path}")

if __name__ == "__main__":
    main()
