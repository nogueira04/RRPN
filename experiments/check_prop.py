import pickle
import numpy as np
import os

proposal_file = "/clusterlivenfs/gnmp/RRPN/data/nucoco/proposals/rfdetr/proposals_train.pkl"
target_id_to_check = 17603 # The ID from the error message

try:
    with open(proposal_file, 'rb') as f:
        proposals = pickle.load(f)
    print(f"Loaded proposals from {proposal_file}")

    if 'ids' not in proposals:
         print("Error: 'ids' key not found in proposal file.")
    else:
         proposal_ids = set(proposals['ids'])
         print(f"Total IDs in proposal file: {len(proposal_ids)}")
         if target_id_to_check in proposal_ids:
             print(f"ID {target_id_to_check} WAS FOUND in proposal IDs.")
         else:
             print(f"ID {target_id_to_check} WAS NOT FOUND in proposal IDs.")

         # Optional: Compare number of proposal IDs with number of images in JSON
         import json
         ann_file = "/clusterlivenfs/gnmp/RRPN/data/nucoco/annotations/rfdetr/output_ann_train/annotations/instances_train.json" # Adjust path if needed
         with open(ann_file, 'r') as f_ann:
             coco_data = json.load(f_ann)
         num_images_in_json = len(coco_data.get('images', []))
         print(f"Total images in JSON file ({os.path.basename(ann_file)}): {num_images_in_json}")
         if len(proposal_ids) != num_images_in_json:
             print(f"*** WARNING: Number of proposal IDs ({len(proposal_ids)}) does not match number of images in JSON ({num_images_in_json})! ***")


except FileNotFoundError:
    print(f"Error: Proposal file not found: {proposal_file}")
except Exception as e:
    print(f"Error loading or checking proposal file: {e}")