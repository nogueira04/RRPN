import argparse
import json
import pickle
import os

def load_ids_from_json(json_path):
    """Loads image IDs from a COCO annotation JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Annotation file not found: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Ensure 'images' key exists and is a list
    if 'images' not in data or not isinstance(data['images'], list):
        raise ValueError(f"Invalid COCO format: 'images' key missing or not a list in {json_path}")
    image_ids = {img['id'] for img in data['images']}
    print(f"Loaded {len(image_ids)} unique image IDs from {os.path.basename(json_path)}")
    return image_ids

def load_ids_from_pickle(pickle_path):
    """Loads image IDs from a proposal pickle file."""
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Proposal file not found: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        proposals = pickle.load(f)
    # Ensure 'ids' key exists and is iterable
    if 'ids' not in proposals or not hasattr(proposals['ids'], '__iter__'):
        raise ValueError(f"Invalid proposal format: 'ids' key missing or not iterable in {pickle_path}")
    proposal_ids = set(proposals['ids'])
    print(f"Loaded {len(proposal_ids)} unique image IDs from {os.path.basename(pickle_path)}")
    return proposal_ids

def main():
    parser = argparse.ArgumentParser(description="Verify proposal coverage for a specific dataset split.")
    parser.add_argument('--ann_file', required=True, help='Path to the COCO annotation JSON file for the split.')
    parser.add_argument('--proposal_file', required=True, help='Path to the corresponding proposal pickle file for the split.')
    parser.add_argument('--split_name', required=True, help='Name of the split (e.g., night, rain, other) for reporting.')

    args = parser.parse_args()

    print(f"\n--- Checking Split: {args.split_name} ---")

    try:
        annotation_ids = load_ids_from_json(args.ann_file)
        proposal_ids = load_ids_from_pickle(args.proposal_file)

        missing_in_proposals = annotation_ids - proposal_ids
        extra_in_proposals = proposal_ids - annotation_ids

        if not missing_in_proposals:
            print(f"[OK] All {len(annotation_ids)} images in {os.path.basename(args.ann_file)} have entries in {os.path.basename(args.proposal_file)}.")
        else:
            print(f"[WARNING] {len(missing_in_proposals)} images defined in {os.path.basename(args.ann_file)} are MISSING from {os.path.basename(args.proposal_file)}:")
            # Print first few missing IDs for quick check
            print(f"  Missing IDs (examples): {list(missing_in_proposals)[:20]}")
            # Optionally save all missing IDs to a file
            # with open(f"missing_ids_{args.split_name}.txt", "w") as f:
            #     for img_id in missing_in_proposals:
            #         f.write(str(img_id) + "\n")
            # print(f"  Full list saved to missing_ids_{args.split_name}.txt")


        if extra_in_proposals:
            print(f"[INFO] {len(extra_in_proposals)} images found in {os.path.basename(args.proposal_file)} are NOT defined in {os.path.basename(args.ann_file)} (might be okay if proposals contain extra images).")
            print(f"  Extra IDs (examples): {list(extra_in_proposals)[:10]}")

    except (FileNotFoundError, ValueError, pickle.UnpicklingError) as e:
        print(f"[ERROR] Failed to process files for split '{args.split_name}': {e}")

if __name__ == "__main__":
    main()