import json
import argparse
import os
from tqdm import tqdm
import sys # Import sys for exit

# --- Configuration ---
NUSCENES_TOKEN_KEY = "nusc_token"
EXPECTED_TOKEN_LENGTH = 32
# ---

def check_coco_json_structure(json_file_path):
    """
    Checks the structure of the 'images' list in a COCO JSON file,
    specifically looking for the 'other' field containing 'nusc_token'
    and counts unique valid tokens found.

    Args:
        json_file_path (str): Path to the COCO JSON annotation file.
    """
    if not os.path.exists(json_file_path):
        print(f"Error: File not found at {json_file_path}")
        sys.exit(1) # Exit if file not found

    print(f"Checking file: {json_file_path}")
    print(f"Looking for structure: images[*]['other']['{NUSCENES_TOKEN_KEY}'] with length {EXPECTED_TOKEN_LENGTH}")

    total_images = 0
    with_other_field = 0
    other_is_dict = 0
    with_token_key = 0
    with_valid_token_entry = 0
    malformed_token_count = 0
    unique_valid_tokens = set()
    image_list = [] # Initialize image_list

    try:
        print("Loading JSON file (this may take time for large files)...")
        with open(json_file_path, 'r') as f:
            data = json.load(f) # Load data inside the try block
        print("JSON loaded.")

        if 'images' not in data or not isinstance(data['images'], list):
            print("Error: 'images' key not found or is not a list in the JSON structure.")
            sys.exit(1) # Exit if structure is wrong

        image_list = data['images'] # Assign image_list ONLY if loading succeeds
        total_images = len(image_list)
        print(f"Found {total_images} total image entries. Analyzing structure...")

        for img_entry in tqdm(image_list, desc="Checking Image Entries"):
            other_field = img_entry.get('other')

            if other_field is not None:
                with_other_field += 1
                if isinstance(other_field, dict):
                    other_is_dict += 1
                    if NUSCENES_TOKEN_KEY in other_field:
                        with_token_key += 1
                        token = other_field[NUSCENES_TOKEN_KEY]
                        if isinstance(token, str) and len(token) == EXPECTED_TOKEN_LENGTH:
                            with_valid_token_entry += 1
                            unique_valid_tokens.add(token)
                        else:
                            malformed_token_count +=1

    except FileNotFoundError: # Should have been caught earlier, but redundant check
        print(f"Error: File not found at {json_file_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {json_file_path}. File might be corrupted.")
        sys.exit(1)
    except MemoryError:
        print(f"Error: MemoryError! The JSON file '{json_file_path}' is too large to load into memory.")
        print("Consider using a library like 'ijson' for iterative parsing if this file is extremely large.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during loading or processing: {e}")
        sys.exit(1) # Exit on other errors

    # --- Report Results ---
    # This section will only be reached if JSON loading and initial parsing succeeded
    print("\n--- Analysis Results ---")
    print(f"Total image entries analyzed: {total_images}")
    print(f"Entries with an 'other' field: {with_other_field}")
    if total_images > 0: print(f"  ({with_other_field / total_images * 100:.2f}%)")
    print(f"Entries where 'other' field is a dictionary: {other_is_dict}")
    print(f"Entries with '{NUSCENES_TOKEN_KEY}' key within 'other' dict: {with_token_key}")
    print(f"Entries with a valid '{NUSCENES_TOKEN_KEY}' (string, length {EXPECTED_TOKEN_LENGTH}): {with_valid_token_entry}")
    if with_token_key > 0: print(f"  ({with_valid_token_entry / with_token_key * 100:.2f}% of those with the key)")
    if malformed_token_count > 0: print(f"Entries with '{NUSCENES_TOKEN_KEY}' but invalid value: {malformed_token_count}")
    print(f"Number of UNIQUE valid '{NUSCENES_TOKEN_KEY}' values found: {len(unique_valid_tokens)}")


    print(f"\n--- Target Structure Count ---")
    print(f"Number of images with the expected structure: {with_valid_token_entry}")

    if with_valid_token_entry == total_images and total_images > 0:
        print("\n>>> NOTE: All image entries have the correct token structure.")
    elif with_valid_token_entry > 0:
         print(f"\n>>> NOTE: Found {with_valid_token_entry} entries with the correct token structure, but {total_images - with_valid_token_entry} entries are missing it.")
    else:
         print("\n>>> NOTE: Could not find any image entries with the complete expected token structure.")

    if len(unique_valid_tokens) == with_valid_token_entry and total_images > 0 and with_valid_token_entry > 0 : # Check unique vs valid
         print(">>> SUCCESS: All valid tokens found appear to be unique.")
    elif len(unique_valid_tokens) < with_valid_token_entry:
         print(f"\n>>> PROBLEM DETECTED: Found {with_valid_token_entry} valid token entries, but only {len(unique_valid_tokens)} UNIQUE tokens.")
         print("    This means tokens are being repeated across different image entries in the JSON.")
         print(f"    This explains why the inference script built a map of size {len(unique_valid_tokens)}.")
         print("    The issue likely lies in the generation of the JSON file (`nuscenes_to_coco_combined.py`).")
    elif with_valid_token_entry == 0: # No valid tokens found
         print(">>> FAILURE: No valid tokens found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check COCO JSON 'images' structure and count unique 'nusc_token' values.")
    parser.add_argument("json_file", help="Path to the COCO JSON annotation file.")
    args = parser.parse_args()

    check_coco_json_structure(args.json_file)