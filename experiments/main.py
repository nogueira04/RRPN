import os
import subprocess
import argparse

def run_step_0(nusc_dir, out_dir, split):
    if not os.path.exists(nusc_dir):
        print(f"ERROR: Dataset not found at {nusc_dir}")
        return
 
    if os.path.exists(f"{out_dir}/{split}"):
        rerun = input(f"INFO: Step 1 already completed. Output folder {out_dir}/{split} exists. Do you want to re-run this step? (y/n): ")
        if rerun.lower() != "y":
            return

    print("INFO: Running Step 0 - Convert nuScenes to COCO format...")
    subprocess.run([
        "bash", "0_nuscenes_to_coco.sh",
        "--nusc_dir", nusc_dir,
        "--out_dir", out_dir,
        "--split", split
    ])
    print("INFO: Step 0 completed.")


def run_step_1(ann_file, imgs_dir, out_file, split):
    if not os.path.exists(ann_file) or not os.path.exists(imgs_dir):
        print(f"ERROR: Required files not found. Ensure {ann_file} and {imgs_dir} exist.")
        return
    
    if os.path.exists(f"{out_file}/proposals/proposals_{split}.pkl"):
        rerun = input(f"INFO: Step 1 already completed. Output file {out_file}/proposals/proposals_{split}.pkl exists. Do you want to re-run this step? (y/n): ")
        if rerun.lower() != "y":
            return

    print("INFO: Running Step 1 - Generate RRPN proposals...")
    subprocess.run([
        "bash", "1_generate_proposals.sh",
        "--ann_file", ann_file,
        "--imgs_dir", imgs_dir,
        "--out_file", out_file,
        "--split", split
    ])
    print("INFO: Step 1 completed.")


def run_step_2():
    print("INFO: Running Step 2 - Train model...")
    subprocess.run(["bash", "2_train.sh"])
    print("INFO: Step 2 completed.")


def run_step_3():
    print("INFO: Running Step 3 - Test trained model...")
    subprocess.run(["bash", "3_test.sh"])
    print("INFO: Step 3 completed.")


def run_step_4():
    print("INFO: Running Step 4 - Inference on a single image...")
    subprocess.run(["bash", "4_inference.sh"])
    print("INFO: Step 4 completed.")

def display_menu():
    print("\nMenu:")
    print("1. Step 0 - Convert nuScenes to COCO format")
    print("2. Step 1 - Generate RRPN proposals")
    print("3. Step 2 - Train model")
    print("4. Step 3 - Test trained model")
    print("5. Step 4 - Inference on a single image")
    print("0. Exit")

def main():
    nusc_dir = "/clusterlivenfs/shared_datasets/nuscenes"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root_dir, "data/nucoco")

    parser = argparse.ArgumentParser()
    parser.add_argument("--nusc_dir", type=str, default=nusc_dir)
    args = parser.parse_args()
    nusc_dir = args.nusc_dir

    while True:
        display_menu()
        choice = input("Select a step to execute (0 to exit): ").strip()

        if choice == "1":
            split = input("Enter dataset split (train or val, default 'val'): ").strip() or "val"
            run_step_0(nusc_dir, out_dir, split)
        elif choice == "2":
            split = input("Enter dataset split (train or val, default 'val'): ").strip() or "val"
            ann_file = os.path.join(out_dir, f"annotations/instances_{split}.json")
            imgs_dir = os.path.join(out_dir, split)
            out_file = os.path.join(out_dir, f"proposals/proposals_{split}.pkl")
            run_step_1(ann_file, imgs_dir, out_file, split)
        elif choice == "3":
            run_step_2()
        elif choice == "4":
            run_step_3()
        elif choice == "5":
            run_step_4()
        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid choice, select a valid step.")


if __name__ == "__main__":
    main()
