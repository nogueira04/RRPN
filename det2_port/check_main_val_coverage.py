import argparse
import json
import pickle
import os
from functools import reduce # For set union

# --- Reuse helper functions from script 1 ---
def load_ids_from_json(json_path):
    """Loads image IDs from a COCO annotation JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Annotation file not found: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
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
    if 'ids' not in proposals or not hasattr(proposals['ids'], '__iter__'):
        raise ValueError(f"Invalid proposal format: 'ids' key missing or not iterable in {pickle_path}")
    proposal_ids = set(proposals['ids'])
    print(f"Loaded {len(proposal_ids)} unique image IDs from {os.path.basename(pickle_path)}")
    return proposal_ids
# --- End of reused functions ---

def main():
    parser = argparse.ArgumentParser(description="Verify if the union of split proposals covers the main validation set.")
    parser.add_argument('--main_val_ann_file', required=True, help='Path to the main COCO validation annotation JSON file (e.g., instances_val.json).')
    parser.add_argument('--proposals_night', required=True, help='Path to the night proposal pickle file.')
    parser.add_argument('--proposals_rain', required=True, help='Path to the rain proposal pickle file.')
    parser.add_argument('--proposals_other', required=True, help='Path to the other (complement) proposal pickle file.')

    args = parser.parse_args()

    print("\n--- Checking Main Validation Set Coverage ---")

    try:
        # Load IDs from the main validation annotation file
        main_val_ids = load_ids_from_json(args.main_val_ann_file)

        # Load IDs from all proposal files
        proposal_paths = {
            'night': args.proposals_night,
            'rain': args.proposals_rain,
            'other': args.proposals_other
        }
        proposal_id_sets = {}
        for name, path in proposal_paths.items():
            proposal_id_sets[name] = load_ids_from_pickle(path)

        # Calculate the union of all proposal IDs
        # Ensure no duplicates across proposal files if an image somehow fits multiple categories (unlikely but possible)
        all_proposal_ids = reduce(lambda a, b: a.union(b), proposal_id_sets.values())
        print(f"\nTotal unique image IDs across all proposal files: {len(all_proposal_ids)}")

        # --- Comparisons ---
        missing_from_all_proposals = main_val_ids - all_proposal_ids
        extra_in_proposals_overall = all_proposal_ids - main_val_ids

        # Check 1: Are there images in the main validation set missing from ALL proposals?
        if not missing_from_all_proposals:
            print(f"[OK] All {len(main_val_ids)} images in {os.path.basename(args.main_val_ann_file)} are present in at least one proposal file.")
        else:
            print(f"[ERROR] {len(missing_from_all_proposals)} images from {os.path.basename(args.main_val_ann_file)} are MISSING from ALL proposal files!")
            print(f"  These images will likely cause 'proposal not found' errors during inference.")
            print(f"  Missing IDs (examples): {list(missing_from_all_proposals)[:20]}")
            # Optionally save
            # with open("missing_from_all_proposals.txt", "w") as f:
            #     for img_id in missing_from_all_proposals:
            #         f.write(str(img_id) + "\n")
            # print(f"  Full list saved to missing_from_all_proposals.txt")


        # Check 2: Are there images in the proposals that are not in the main validation set?
        if not extra_in_proposals_overall:
            print(f"[OK] All {len(all_proposal_ids)} images found in proposal files are also present in {os.path.basename(args.main_val_ann_file)}.")
        else:
            print(f"[INFO] {len(extra_in_proposals_overall)} images found across proposal files are NOT in {os.path.basename(args.main_val_ann_file)}.")
            print(f"  This might indicate proposals were generated on a slightly different set or includes images not in the final val set.")
            print(f"  Extra IDs (examples): {list(extra_in_proposals_overall)[:10]}")

        # Optional: Check for overlaps between proposal sets (sanity check)
        overlap_night_rain = proposal_id_sets['night'].intersection(proposal_id_sets['rain'])
        overlap_night_other = proposal_id_sets['night'].intersection(proposal_id_sets['other'])
        overlap_rain_other = proposal_id_sets['rain'].intersection(proposal_id_sets['other'])

        if overlap_night_rain or overlap_night_other or overlap_rain_other:
            print("[WARNING] Overlap detected between proposal sets:")
            if overlap_night_rain: print(f"  - Night & Rain overlap: {len(overlap_night_rain)} images")
            if overlap_night_other: print(f"  - Night & Other overlap: {len(overlap_night_other)} images")
            if overlap_rain_other: print(f"  - Rain & Other overlap: {len(overlap_rain_other)} images")
            print("  This indicates some images might be ambiguously categorized by your splitting criteria or scene mapping.")
        else:
            print("[OK] No overlap found between the different proposal sets (night, rain, other).")


    except (FileNotFoundError, ValueError, pickle.UnpicklingError) as e:
        print(f"[ERROR] Failed processing files: {e}")

if __name__ == "__main__":
    main()