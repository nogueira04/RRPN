import os
import copy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

try:
    from pynuscenes.nuscenes_dataset import NuscenesDataset
    PYNUSCENES_AVAILABLE = True
except ImportError:
    print("ERROR: Failed to import NuscenesDataset from pynuscenes.")
    print("Please ensure pynuscenes is installed: pip install pynuscenes")
    PYNUSCENES_AVAILABLE = False
    class NuscenesDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError("pynuscenes.NuscenesDataset is required for this script's ordering logic.")
        def __len__(self): return 0
        def __getitem__(self, idx): return None

DEFAULT_NUSC_ROOT = '../data/nuscenes'
DEFAULT_SPLIT = 'mini_val'
DEFAULT_CAMERAS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
DEFAULT_OUTPUT_DIR = 'nuscenes_gt_3d_vis_ordered'
MAX_IMAGES_TO_SAVE = 50

BOX_COLOR_RGB = (1, 0, 0)
FRONT_BOX_COLOR_RGB = (0, 0, 1)
BOX_LINEWIDTH = 1.5

def draw_projected_box3d(ax, corners_2d, color=(1,0,0), linewidth=1.5,
                         front_color=(0,0,1), front_linewidth_scale=1.2):
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    front_face_edge_pairs = [(0, 1), (1, 5), (5, 4), (4, 0)]
    for i, j in edges:
        current_color = color
        current_linewidth = linewidth
        is_front_edge = any(((i == fp_i and j == fp_j) or (i == fp_j and j == fp_i)) for fp_i, fp_j in front_face_edge_pairs)
        if is_front_edge:
            current_color = front_color
            current_linewidth *= front_linewidth_scale
        ax.plot([corners_2d[0, i], corners_2d[0, j]],
                [corners_2d[1, i], corners_2d[1, j]],
                color=current_color, linewidth=current_linewidth)

def parse_args():
    parser = argparse.ArgumentParser(description='Visualizes NuScenes GT 3D annotations.')
    parser.add_argument('--nusc_root', default=DEFAULT_NUSC_ROOT, help='NuScenes dataroot path.')
    parser.add_argument('--split', default=DEFAULT_SPLIT, choices=['mini_train', 'mini_val', 'train', 'val', 'test'], help='NuScenes split.')
    parser.add_argument('--out_dir', default=DEFAULT_OUTPUT_DIR, help='Output directory.')
    parser.add_argument('--cameras', nargs='+', default=DEFAULT_CAMERAS, help='Cameras to process.')
    parser.add_argument('--max_images', type=int, default=MAX_IMAGES_TO_SAVE, help='Max images to save.')
    parser.add_argument('--nsweeps_radar', default=1, type=int, help='(for pynuscenes compatibility).')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if not PYNUSCENES_AVAILABLE:
        print("Exiting due to missing pynuscenes.NuscenesDataset.")
        return

    print(f"Visualizing NuScenes GT for split: {args.split}, up to {args.max_images} images to: {args.out_dir}")
    print(f"Processing cameras: {args.cameras}")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"Created output directory: {args.out_dir}")

    if "mini" in args.split: nusc_version = "v1.0-mini"
    elif "test" in args.split: nusc_version = "v1.0-test"
    else: nusc_version = "v1.0-trainval"

    print(f"Loading NuscenesDataset (pynuscenes): {nusc_version}, split '{args.split}'...")
    try:
        dataset_for_ordering = NuscenesDataset(nusc_path=args.nusc_root,
                                               nusc_version=nusc_version,
                                               split=args.split,
                                               coordinates='vehicle',
                                               nsweeps_radar=args.nsweeps_radar,
                                               sensors_to_return=['camera'],
                                               pc_mode='camera',
                                               logging_level='WARNING')
        nusc_api = dataset_for_ordering.nusc
        print(f"NuscenesDataset loaded with {len(dataset_for_ordering)} samples for ordering.")
    except Exception as e:
        print(f"Critical Error: Failed to load NuscenesDataset: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Building filename to sample_data_token map...")
    filename_to_sd_token_map = {}
    if hasattr(nusc_api, 'sample_data') and nusc_api.sample_data:
        for sd_record in tqdm(nusc_api.sample_data, desc="Mapping filenames"):
            filename_to_sd_token_map[sd_record['filename']] = sd_record['token'] # Key is relative path
    else:
        print("Error: nusc_api.sample_data is not available or empty. Cannot build filename map.")
        return
    print(f"Filename map built with {len(filename_to_sd_token_map)} entries.")
    # print(f"DEBUG: Example key from filename_to_sd_token_map: {list(filename_to_sd_token_map.keys())[0] if filename_to_sd_token_map else 'Map is empty'}")


    saved_image_count = 0
    output_image_idx = 0
    num_ordered_samples = len(dataset_for_ordering)
    print(f"Iterating through {num_ordered_samples} samples from NuscenesDataset...")

    for i in tqdm(range(num_ordered_samples), desc=f"Visualizing {args.split}"):
        if saved_image_count >= args.max_images:
            print(f"Reached maximum images to save ({args.max_images}). Stopping.")
            break
        try:
            sample_from_pynusc = dataset_for_ordering[i]
            if sample_from_pynusc is None:
                print(f"Warning: NuscenesDataset returned None for index {i}. Skipping.")
                continue

            for cam_info_pynusc in sample_from_pynusc.get('camera', []):
                if saved_image_count >= args.max_images: break

                cam_name = cam_info_pynusc.get('camera_name')
                if not cam_name or cam_name not in args.cameras:
                    continue

                # --- TOKEN EXTRACTION USING FILENAME MAP (WITH PATH NORMALIZATION) ---
                sample_data_token = None
                abs_cam_filename_from_pynusc = cam_info_pynusc.get('cam_path')

                if abs_cam_filename_from_pynusc:
                    relative_cam_filename = None
                    try:
                        # Ensure nusc_api.dataroot is an absolute path itself for relpath to work reliably
                        # and that it's the *correct* dataroot for the paths pynuscenes generates.
                        # If args.nusc_root was relative, nusc_api.dataroot might also be relative.
                        # It's best if nusc_api.dataroot is absolute and normalized.
                        dataroot_abs = os.path.abspath(nusc_api.dataroot)
                        abs_cam_filename_norm = os.path.abspath(abs_cam_filename_from_pynusc)

                        if not abs_cam_filename_norm.startswith(dataroot_abs):
                             print(f"Warning: Normalized pynuscenes path '{abs_cam_filename_norm}' does not start with "
                                   f"normalized dataroot '{dataroot_abs}'. Cannot reliably get relative path. Skipping.")
                        else:
                            relative_cam_filename = os.path.relpath(abs_cam_filename_norm, dataroot_abs)
                            # NuScenes filenames in DB don't have leading slashes if they are truly relative to dataroot
                            if relative_cam_filename.startswith(os.sep):
                                relative_cam_filename = relative_cam_filename[len(os.sep):]

                    except ValueError as e_relpath: # os.path.relpath can raise ValueError
                        print(f"Warning: os.path.relpath failed for '{abs_cam_filename_from_pynusc}' relative to '{nusc_api.dataroot}': {e_relpath}. Skipping.")

                    if relative_cam_filename:
                        sample_data_token = filename_to_sd_token_map.get(relative_cam_filename)
                        if not sample_data_token:
                            print(f"Warning: Relative filename '{relative_cam_filename}' (derived from pynuscenes abs_path '{abs_cam_filename_from_pynusc}') "
                                  f"not found in map. Skipping camera {cam_name} in sample index {i}.")
                            # For a quick check if you're still having issues:
                            # if i % 50 == 0: # Print occasionally
                            #     print(f"    DEBUG: Looking for: '{relative_cam_filename}'")
                            #     if filename_to_sd_token_map:
                            #         print(f"    DEBUG: Example map key: '{list(filename_to_sd_token_map.keys())[0]}'")
                            #     else:
                            #         print(f"    DEBUG: filename_to_sd_token_map is empty!")
                            continue
                    else:
                        # This means conversion to relative path failed or was deemed unsafe
                        if abs_cam_filename_from_pynusc: # Only print if we had a path to begin with
                            print(f"Warning: Failed to obtain a valid relative path from '{abs_cam_filename_from_pynusc}'. Skipping camera {cam_name} in sample index {i}.")
                        continue # Skip if relative_cam_filename is None or empty
                else:
                    print(f"Warning: Missing 'cam_path' in cam_info_pynusc for camera {cam_name}, "
                          f"sample index {i}. Cannot retrieve sample_data_token. Skipping.")
                    if isinstance(cam_info_pynusc, dict):
                        print(f"DEBUG: Keys in cam_info_pynusc that led to this: {list(cam_info_pynusc.keys())}")
                    continue
                # --- END TOKEN EXTRACTION ---


                try:
                    cam_sample_data = nusc_api.get('sample_data', sample_data_token)
                except KeyError:
                    print(f"Error: sample_data_token '{sample_data_token}' (from map) not found in nusc_api's sample_data table. This is unexpected.")
                    continue

                image_path = os.path.join(nusc_api.dataroot, cam_sample_data['filename'])
                try:
                    img_pil = Image.open(image_path).convert("RGB")
                    img_width, img_height = img_pil.size
                except FileNotFoundError:
                    print(f"Warning: Image file not found: {image_path}. Skipping.")
                    continue
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}. Skipping.")
                    continue

                parent_sample_token = cam_sample_data['sample_token']
                parent_sample_record = nusc_api.get('sample', parent_sample_token)

                fig, ax = plt.subplots(1, 1, figsize=(img_width / 100, img_height / 100), dpi=100)
                ax.imshow(img_pil)
                ax.axis('off')

                ax.set_xlim(0, img_width)
                ax.set_ylim(img_height, 0) # Matplotlib's imshow typically inverts y-axis
                ax.set_aspect('equal', adjustable='box') # Crucial for image display

                cs_record_gt = nusc_api.get('calibrated_sensor', cam_sample_data['calibrated_sensor_token'])
                cam_intrinsic_gt = np.array(cs_record_gt['camera_intrinsic'])
                ego_pose_record_gt = nusc_api.get('ego_pose', cam_sample_data['ego_pose_token'])
                ego_translation_glob = np.array(ego_pose_record_gt['translation'])
                ego_rotation_glob = Quaternion(ego_pose_record_gt['rotation'])
                sensor_translation_ego = np.array(cs_record_gt['translation'])
                sensor_rotation_ego = Quaternion(cs_record_gt['rotation'])

                for ann_token in parent_sample_record['anns']:
                    ann_record = nusc_api.get('sample_annotation', ann_token)
                    box_global = Box(ann_record['translation'], ann_record['size'], Quaternion(ann_record['rotation']),
                                     name=ann_record['category_name'], token=ann_record['token'])
                    box_cam = copy.deepcopy(box_global)
                    box_cam.translate(-ego_translation_glob)
                    box_cam.rotate(ego_rotation_glob.inverse)
                    box_cam.translate(-sensor_translation_ego)
                    box_cam.rotate(sensor_rotation_ego.inverse)

                    if not np.any(box_cam.corners()[2, :] > 0.1): continue
                    corners_3d_cam = box_cam.corners()
                    corners_2d_img = view_points(corners_3d_cam, cam_intrinsic_gt, normalize=True)[:2, :]
                    if box_in_image(box_cam, cam_intrinsic_gt, (img_width, img_height), vis_level=BoxVisibility.ANY):
                        draw_projected_box3d(ax, corners_2d_img, color=BOX_COLOR_RGB, linewidth=BOX_LINEWIDTH,
                                             front_color=FRONT_BOX_COLOR_RGB, front_linewidth_scale=1.2)

                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                output_filename = os.path.join(args.out_dir, f"{args.split}_{output_image_idx:05d}_{cam_name}_{sample_data_token}.png")
                plt.savefig(output_filename, dpi=100, pad_inches=0)
                plt.close(fig)
                saved_image_count += 1
                output_image_idx += 1
        except KeyboardInterrupt:
            print("KeyboardInterrupt. Stopping.")
            break
        except Exception as e:
            print(f"Error processing pynuscenes sample index {i} (pynusc cam_path: {cam_info_pynusc.get('cam_path', 'N/A')}): {e}")
            import traceback
            traceback.print_exc()
            if saved_image_count >= args.max_images: break
            continue

    print(f"\nFinished. Saved {saved_image_count} images to {args.out_dir}")

if __name__ == '__main__':
    if not PYNUSCENES_AVAILABLE:
        print("pynuscenes.NuscenesDataset is required.")
    else:
        main()