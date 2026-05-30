import pickle
import argparse
import os
import sys
from collections import Counter

def main():
    parser = argparse.ArgumentParser(description="Check the contents and scene type distribution of an id_to_scene .pkl file.")
    parser.add_argument("pkl_file", help="Path to the .pkl file (e.g., id_to_scene_val.pkl)")
    args = parser.parse_args()

    file_path = args.pkl_file
    print(f"Checking file: {file_path}")

    # --- Validate File Path ---
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
    if not os.path.isfile(file_path):
         print(f"Error: Path provided is not a file: {file_path}")
         sys.exit(1)

    # --- Load the Pickle File ---
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except pickle.UnpicklingError:
        print(f"Error: Could not unpickle the file. It might be corrupted or not a valid pickle file.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # --- Validate Loaded Data ---
    if not isinstance(data, dict):
        print(f"Error: Expected a dictionary inside the pickle file, but found type {type(data)}.")
        sys.exit(1)

    if not data:
        print("The dictionary in the pickle file is empty.")
        sys.exit(0) # Exit normally, file is valid but empty

    # --- Analyze the Data ---
    total_entries = len(data)
    print(f"\nSuccessfully loaded dictionary with {total_entries} entries.")

    # Check a sample key-value pair
    sample_key = next(iter(data)) # Get the first key
    sample_value = data[sample_key]
    print(f"Sample entry: Image ID {sample_key} (type: {type(sample_key)}) -> Scene Type '{sample_value}' (type: {type(sample_value)})")

    # Count scene types
    scene_types = list(data.values()) # Get all the scene type strings
    scene_counts = Counter(scene_types)
    unique_types = len(scene_counts)

    print(f"\nFound {unique_types} unique scene type(s):")
    print("-" * 20)
    # Sort by count descending for better readability
    for scene_type, count in scene_counts.most_common():
        percentage = (count / total_entries) * 100
        print(f"  - '{scene_type}': {count} entries ({percentage:.2f}%)")
    print("-" * 20)

if __name__ == "__main__":
    main()