import json
import argparse
import sys

def check_invalid_categories(ann_file_path, invalid_id=-1):
    """
    Checks a COCO JSON annotation file for annotations with a specific invalid category ID.

    Args:
        ann_file_path (str): Path to the COCO JSON annotation file.
        invalid_id (int): The category ID to check for (default is -1).

    Returns:
        list: A list of annotation IDs that have the invalid category ID.
              Returns None if the file cannot be read or parsed.
    """
    invalid_ann_ids = []
    try:
        with open(ann_file_path, 'r') as f:
            data = json.load(f)
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
    if not annotations:
        print("No 'annotations' list found in the JSON file.")
        return [] # Return empty list if no annotations section

    print(f"Checking {len(annotations)} annotations for category_id == {invalid_id}...")

    for ann in annotations:
        # Check if 'category_id' exists and equals the invalid_id
        if ann.get('category_id') == invalid_id:
            ann_id = ann.get('id', 'N/A') # Get annotation ID if available
            img_id = ann.get('image_id', 'N/A') # Get image ID if available
            print(f"  Found annotation with id={ann_id} (image_id={img_id}) having category_id={invalid_id}")
            invalid_ann_ids.append(ann_id)

    return invalid_ann_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check COCO annotation file for invalid category IDs.')
    parser.add_argument('ann_file', help='Path to the COCO annotation JSON file.')
    parser.add_argument('--invalid_id', type=int, default=-1,
                        help='The invalid category ID to search for (default: -1).')

    args = parser.parse_args()

    invalid_ids_found = check_invalid_categories(args.ann_file, args.invalid_id)

    if invalid_ids_found is None:
        sys.exit(1) # Exit with error code if file reading failed

    if invalid_ids_found:
        print(f"\nCheck finished: Found {len(invalid_ids_found)} annotations with category_id = {args.invalid_id}.")
    else:
        print(f"\nCheck finished: No annotations found with category_id = {args.invalid_id}.")