import cv2
import random
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
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

category_id_to_name = {0: '_', 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "bus", 6: "truck"}

def register_datasets():
    for dataset_name, dataset_info in _DATASETS.items():
        # Register the dataset
        DatasetCatalog.register(dataset_name, lambda dataset_info=dataset_info: load_coco_json(dataset_info['ann_file'], dataset_info['img_dir']))
        # Set the metadata for the dataset (e.g., class names)
        MetadataCatalog.get(dataset_name).set(thing_classes=list(category_id_to_name.values()))

DatasetCatalog.clear()  # Clear any existing dataset registration
MetadataCatalog.clear()  # Clear existing metadata
register_datasets()
# MetadataCatalog.get("nucoco_val").set(thing_classes=list(category_id_to_name.values()))
# Load the dataset
dataset_name = "nucoco_train"  # Change to your dataset name if different
dataset_dicts = DatasetCatalog.get(dataset_name)
metadata = MetadataCatalog.get(dataset_name)

def load_gt():
    # Visualize a few random samples
    for i in range(50):  # Visualize 5 random images        
        d = random.choice(dataset_dicts)
        img = cv2.imread(d["file_name"])
        
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.2)
        vis = visualizer.draw_dataset_dict(d)
        
        # Convert from RGB to BGR for OpenCV display
        img_with_gt = vis.get_image()[:, :, ::-1]
        
        
        # Save the image with ground truth bounding boxes if desired
        cv2.imwrite(f"ground_truth_{i+1}.jpg", img_with_gt)

def check_idx():
    import json

    # Load your COCO-style JSON annotation file
    annotation_file = '/clusterlivenfs/gnmp/RRPN/data/nucoco/annotations/instances_val.json'

    with open(annotation_file) as f:
        coco_data = json.load(f)

    # Extract all category IDs in the dataset
    category_ids = set()
    for ann in coco_data['annotations']:
        category_ids.add(ann['category_id'])

    print("Unique category IDs in annotations:", sorted(category_ids))

    # Now check the mapping of these IDs to their class names
    category_id_to_name = {0: '_', 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "bus", 6: "truck"}

    # Print the class name for each category ID found
    for cat_id in sorted(category_ids):
        class_name = category_id_to_name.get(cat_id, "Unknown Class ID")
        print(f"Category ID: {cat_id}, Class Name: {class_name}")

def get_ann_classes():
    import json

    # Load your COCO-style JSON annotation file
    annotation_file = '/clusterlivenfs/gnmp/RRPN/data/nucoco/annotations/instances_val.json'

    with open(annotation_file) as f:
        coco_data = json.load(f)

    # Extract all categories
    categories = coco_data.get('categories', [])

    # Create a dictionary to map category IDs to class names
    category_id_to_name = {}
    for category in categories:
        category_id = category['id']
        category_name = category['name']
        category_id_to_name[category_id] = category_name

    # Print all classes
    print("All classes in the annotation file:")
    for category_id, category_name in category_id_to_name.items():
        print(f"Category ID: {category_id}, Class Name: {category_name}")

check_idx()