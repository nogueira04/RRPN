# File: check_coco_ids.py
import json
import argparse
import sys
from collections import Counter

def check_coco_category_ids(ann_file_path, expected_ids=None):
    """
    Checks a COCO JSON annotation file for the range and distribution of category IDs.

    Args:
        ann_file_path (str): Path to the COCO JSON annotation file.
        expected_ids (set, optional): A set of expected category IDs. If provided,
                                      it will check if all found IDs are within this set.

    Returns:
        bool: True if checks pass (or no annotations), False otherwise.
              Returns None if the file cannot be read or parsed.
    """
    all_found_ids = []
    category_counts = Counter()

    try:
        print(f"--- Reading file: {ann_file_path} ---")
        with open(ann_file_path, 'r') as f:
            data = json.load(f)
        print("File read successfully.")
    except FileNotFoundError:
        print(f"Error: Annotation file not found at '{ann_file_path}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON from '{ann_file_path}'. File might be corrupted.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return None

    annotations = data.get('annotations', [])
    if not isinstance(annotations, list):
         print("Error: 'annotations' key does not contain a list.")
         return None

    if not annotations:
        print("No 'annotations' found in the JSON file. Check finished.")
        return True # Technically passes as there are no invalid IDs

    print(f"Checking {len(annotations)} annotations for category IDs...")

    min_id = float('inf')
    max_id = float('-inf')
    valid_check = True

    for i, ann in enumerate(annotations):
        if i % 50000 == 0 and i > 0: # Print progress for large files
             print(f"  Processed {i} annotations...")

        cat_id = ann.get('category_id')

        # Check if category_id exists and is an integer
        if cat_id is None:
            print(f"  Error: Annotation with id={ann.get('id', 'N/A')} (image_id={ann.get('image_id', 'N/A')}) is missing 'category_id'.")
            valid_check = False
            continue # Skip this annotation

        if not isinstance(cat_id, int):
             print(f"  Error: Annotation with id={ann.get('id', 'N/A')} (image_id={ann.get('image_id', 'N/A')}) has non-integer category_id: {cat_id} (type: {type(cat_id)}).")
             valid_check = False
             continue # Skip this annotation

        # Update stats
        all_found_ids.append(cat_id)
        category_counts[cat_id] += 1
        min_id = min(min_id, cat_id)
        max_id = max(max_id, cat_id)

        # Check against expected IDs if provided
        if expected_ids is not None and cat_id not in expected_ids:
            print(f"  Error: Annotation with id={ann.get('id', 'N/A')} (image_id={ann.get('image_id', 'N/A')}) has unexpected category_id: {cat_id}")
            valid_check = False
            # Don't continue here, let it count towards unique IDs

    print("Finished scanning annotations.")

    if not all_found_ids: # Should only happen if all annotations had errors
        print("Warning: No valid category IDs were processed.")
        return valid_check # Return status based on errors found during scan

    unique_ids = sorted(list(set(all_found_ids)))

    print(f"\nMin category_id found: {min_id}")
    print(f"Max category_id found: {max_id}")
    print(f"Unique category_ids found: {unique_ids}")

    print("\nCategory ID Counts:")
    for cat_id in sorted(category_counts.keys()):
        print(f"  ID {cat_id}: {category_counts[cat_id]} annotations")

    if expected_ids is not None:
        print(f"\nExpected category IDs based on train_net.py mapping: {expected_ids}")
        found_set = set(unique_ids)
        if found_set.issubset(expected_ids):
            print("\nAll category IDs found in the JSON are within the expected range.")
        else:
            unexpected = found_set - expected_ids
            print(f"\n*** Error: Found unexpected category IDs: {unexpected} ***")
            valid_check = False

    return valid_check

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check COCO annotation file category IDs.')
    parser.add_argument('ann_file', help='Path to the COCO annotation JSON file.')
    # Add expected IDs based on your train_net.py mapping
    # Make this a list of integers
    parser.add_argument('--expected', type=int, nargs='*', default=[0, 1, 2, 3, 4, 5],
                        help='List of expected integer category IDs (default: 0 1 2 3 4 5).')

    args = parser.parse_args()

    expected_id_set = set(args.expected) if args.expected else None

    check_passed = check_coco_category_ids(args.ann_file, expected_id_set)

    print("\n---")
    if check_passed is None:
        print("Check could not be completed due to file errors.")
        sys.exit(1)
    elif check_passed:
        print("Check completed: No unexpected or invalid category IDs found.")
        sys.exit(0)
    else:
        print("Check completed: Found errors or unexpected category IDs (see details above).")
        sys.exit(1) # Exit with error code if issues found