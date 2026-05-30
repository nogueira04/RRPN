"""Generate paired 3D / 2D-projection visualizations on the same NuScenes sample.

For each selected COCO image, writes two files into --out_dir:
  pair_<image_id>_3d.png  - NuScenes 3D cuboids projected to the image
                            (red edges, blue front face). Same primitive as
                            tools/draw_3d_ann.py.
  pair_<image_id>_2d.png  - axis-aligned COCO bboxes drawn with `supervision`
                            (or OpenCV fallback). Same primitive as
                            tools/check_ann.py.

Selection is automatic: every image is scored by a "projection-problem"
heuristic (overlap clutter + small-inside-large engulfment + truncation),
ranked descending, and the top --count are rendered.
"""

import argparse
import csv
import json
import logging
import os
from itertools import combinations

import matplotlib

matplotlib.use("Agg")  # headless on the cluster

import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image, view_points
from pyquaternion import Quaternion

try:
    import supervision as sv

    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# RRPN / nucoco categories. The COCO JSON only has these 6.
RRPN_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

# NuScenes raw class -> RRPN class (mirrors tools/nuscenes_to_coco.py mapping).
NUSC_TO_RRPN = {
    "human.pedestrian.adult": "person",
    "human.pedestrian.child": "person",
    "human.pedestrian.construction_worker": "person",
    "human.pedestrian.police_officer": "person",
    "vehicle.bicycle": "bicycle",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "truck",
    "vehicle.emergency.ambulance": "truck",
    "vehicle.emergency.police": "car",
}

BOX_COLOR_RGB = (1, 0, 0)
FRONT_BOX_COLOR_RGB = (0, 0, 1)
BOX_LINEWIDTH = 1.5


# --------------------------------------------------------------------------- #
# 3D drawing primitive (verbatim from tools/draw_3d_ann.py:38-56).
# --------------------------------------------------------------------------- #
def draw_projected_box3d(
    ax,
    corners_2d,
    color=(1, 0, 0),
    linewidth=1.5,
    front_color=(0, 0, 1),
    front_linewidth_scale=1.2,
):
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    front_face_edge_pairs = [(0, 1), (1, 5), (5, 4), (4, 0)]
    for i, j in edges:
        is_front = any(
            ((i == fi and j == fj) or (i == fj and j == fi))
            for fi, fj in front_face_edge_pairs
        )
        c = front_color if is_front else color
        lw = linewidth * front_linewidth_scale if is_front else linewidth
        ax.plot(
            [corners_2d[0, i], corners_2d[0, j]],
            [corners_2d[1, i], corners_2d[1, j]],
            color=c,
            linewidth=lw,
        )


# --------------------------------------------------------------------------- #
# 2D drawing primitive (adapted from tools/check_ann.py:182-224 + 137-180).
# --------------------------------------------------------------------------- #
def draw_2d_supervision(image_bgr, anns, cat_id_to_name):
    boxes, class_ids, confidences, labels = [], [], [], []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
        cid = ann["category_id"]
        class_ids.append(cid)
        confidences.append(ann.get("score", 1.0))
        labels.append(cat_id_to_name.get(cid, "UNK"))
    if not boxes:
        return image_bgr
    detections = sv.Detections(
        xyxy=np.array(boxes),
        class_id=np.array(class_ids),
        confidence=np.array(confidences),
    )
    annotated = sv.BoxAnnotator().annotate(
        scene=image_bgr.copy(), detections=detections
    )
    annotated = sv.LabelAnnotator(text_scale=1.15, text_thickness=2).annotate(
        scene=annotated,
        detections=detections,
        labels=labels,
    )
    return annotated


def draw_2d_opencv(image_bgr, anns, cat_id_to_name, color_map):
    img = image_bgr.copy()
    h_img, w_img = img.shape[:2]
    for ann in anns:
        cid = ann["category_id"]
        name = cat_id_to_name.get(cid, "UNK")
        color = color_map.setdefault(
            cid, tuple(np.random.randint(0, 256, size=3).tolist())
        )
        x, y, w, h = map(int, ann["bbox"])
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img - 1, x + w), min(h_img - 1, y + h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img,
            name,
            (x1 + 2, max(y1 - 4, th)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    return img


# --------------------------------------------------------------------------- #
# Problem-detection heuristic.
# --------------------------------------------------------------------------- #
def _iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0, 0.0  # iou, intersection
    union = aw * ah + bw * bh - inter
    return inter / union, inter


def score_image(anns, img_w, img_h, iou_thr, contain_thr, size_frac):
    if not anns:
        return 0.0
    img_area = img_w * img_h
    score = 0.0
    for ann in anns:
        _, _, w, h = ann["bbox"]
        if (w * h) > size_frac * img_area:
            score += 1.0  # truncation-style oversized box
    for a, b in combinations(anns, 2):
        if a["category_id"] != b["category_id"]:
            continue
        iou, inter = _iou(a["bbox"], b["bbox"])
        if iou > iou_thr:
            score += 1.0
        ax_area = a["bbox"][2] * a["bbox"][3]
        bx_area = b["bbox"][2] * b["bbox"][3]
        if ax_area == 0 or bx_area == 0:
            continue
        small_area, large_area = sorted([ax_area, bx_area])
        if (inter / small_area) > contain_thr and (small_area / large_area) < 0.3:
            score += 2.0
    return score


# --------------------------------------------------------------------------- #
# Recover sample_data token for a COCO image.
# The converter (tools/nuscenes_to_coco.py) stores `cam_cs_record.token` under
# `other.nusc_token`, which is actually the *calibrated_sensor* token, not the
# sample_data token. Many sample_data records share the same calibrated_sensor
# (same camera in the same scene), so we disambiguate by pixel-matching the
# COCO JPG against candidate NuScenes camera JPGs.
# --------------------------------------------------------------------------- #
def build_cs_to_camera_sd_index(nusc):
    """cs_token -> list of sample_data records (cameras + keyframes only)."""
    idx = {}
    for sd in nusc.sample_data:
        if sd.get("fileformat") != "jpg":
            continue
        if not sd.get("is_key_frame", False):
            continue
        idx.setdefault(sd["calibrated_sensor_token"], []).append(sd)
    return idx


def match_sd_for_coco_image(nusc, cs_to_sd, coco_img_bgr, cs_token):
    """Given the COCO image bytes and its calibrated_sensor token, return the
    sample_data token whose NuScenes JPG most closely matches."""
    candidates = cs_to_sd.get(cs_token, [])
    if not candidates:
        return None
    coco_small = cv2.resize(coco_img_bgr, (160, 90)).astype(np.int16)
    best_tok, best_diff = None, float("inf")
    for sd in candidates:
        path = os.path.join(nusc.dataroot, sd["filename"])
        nu = cv2.imread(path)
        if nu is None:
            continue
        if nu.shape[:2] != coco_img_bgr.shape[:2]:
            nu = cv2.resize(nu, (coco_img_bgr.shape[1], coco_img_bgr.shape[0]))
        nu_small = cv2.resize(nu, (160, 90)).astype(np.int16)
        diff = float(np.mean(np.abs(coco_small - nu_small)))
        if diff < best_diff:
            best_diff, best_tok = diff, sd["token"]
    return best_tok if best_diff < 25.0 else None  # 25/255 mean-abs-diff cutoff


# --------------------------------------------------------------------------- #
# 3D rendering (math block adapted from tools/draw_3d_ann.py:211-238).
# --------------------------------------------------------------------------- #
def render_3d(nusc, sd_token, out_path, show_all=False):
    sd = nusc.get("sample_data", sd_token)
    sample = nusc.get("sample", sd["sample_token"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])

    cam_intrinsic = np.array(cs["camera_intrinsic"])
    ego_t = np.array(pose["translation"])
    ego_r = Quaternion(pose["rotation"])
    sensor_t = np.array(cs["translation"])
    sensor_r = Quaternion(cs["rotation"])

    image_path = os.path.join(nusc.dataroot, sd["filename"])
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    fig, ax = plt.subplots(1, 1, figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(img)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    for ann_token in sample["anns"]:
        ann_record = nusc.get("sample_annotation", ann_token)
        if not show_all and NUSC_TO_RRPN.get(ann_record["category_name"]) is None:
            continue
        box_global = Box(
            ann_record["translation"],
            ann_record["size"],
            Quaternion(ann_record["rotation"]),
            name=ann_record["category_name"],
            token=ann_record["token"],
        )
        box_cam = copy.deepcopy(box_global)
        box_cam.translate(-ego_t)
        box_cam.rotate(ego_r.inverse)
        box_cam.translate(-sensor_t)
        box_cam.rotate(sensor_r.inverse)

        if not np.any(box_cam.corners()[2, :] > 0.1):
            continue
        corners_3d = box_cam.corners()
        corners_2d = view_points(corners_3d, cam_intrinsic, normalize=True)[:2, :]
        if box_in_image(box_cam, cam_intrinsic, (w, h), vis_level=BoxVisibility.ANY):
            draw_projected_box3d(
                ax,
                corners_2d,
                color=BOX_COLOR_RGB,
                linewidth=BOX_LINEWIDTH,
                front_color=FRONT_BOX_COLOR_RGB,
                front_linewidth_scale=1.2,
            )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, dpi=100, pad_inches=0)
    plt.close(fig)


def render_2d(coco_image_path, anns, cat_id_to_name, out_path, color_map):
    image_bgr = cv2.imread(coco_image_path)
    if image_bgr is None:
        raise FileNotFoundError(coco_image_path)
    if SUPERVISION_AVAILABLE:
        annotated = draw_2d_supervision(image_bgr, anns, cat_id_to_name)
    else:
        annotated = draw_2d_opencv(image_bgr, anns, cat_id_to_name, color_map)
    cv2.imwrite(out_path, annotated)


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ann_file", default="data/nucoco/annotations/instances_val.json")
    p.add_argument("--img_dir", default="data/nucoco/val")
    p.add_argument("--nusc_root", default="/clusterlivenfs/shared_datasets/nuscenes")
    p.add_argument("--nusc_version", default="v1.0-trainval")
    p.add_argument("--out_dir", default="tools/viz_pairs")
    p.add_argument("--count", type=int, default=10, help="Number of pairs to render.")
    p.add_argument("--iou_thr", type=float, default=0.3)
    p.add_argument("--contain_thr", type=float, default=0.9)
    p.add_argument(
        "--size_frac",
        type=float,
        default=0.4,
        help="Box-area / image-area threshold for truncation flag.",
    )
    p.add_argument(
        "--show_all_3d",
        action="store_true",
        help="Draw all NuScenes 3D anns, not just the 6 RRPN categories.",
    )
    p.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Optional comma-separated COCO image IDs to override auto-pick.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not SUPERVISION_AVAILABLE:
        logger.warning(
            "supervision not installed; falling back to plain OpenCV drawing."
        )

    logger.info(f"Loading COCO annotations: {args.ann_file}")
    with open(args.ann_file) as f:
        coco = json.load(f)
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_id_to_info = {im["id"]: im for im in coco["images"]}

    img_id_to_anns = {}
    for ann in coco["annotations"]:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)
    logger.info(
        f"  {len(img_id_to_info)} images, {len(coco['annotations'])} anns, "
        f"{len(cat_id_to_name)} categories."
    )

    if args.ids:
        wanted = [int(x) for x in args.ids.split(",")]
        ranked = []
        for iid in wanted:
            if iid not in img_id_to_info:
                logger.warning(f"image_id {iid} not in COCO file; skipping")
                continue
            im = img_id_to_info[iid]
            anns = img_id_to_anns.get(iid, [])
            s = score_image(
                anns,
                im["width"],
                im["height"],
                args.iou_thr,
                args.contain_thr,
                args.size_frac,
            )
            ranked.append((s, im, anns))
    else:
        logger.info("Scoring all images (problem heuristic)...")
        ranked = []
        for iid, im in tqdm(img_id_to_info.items(), desc="scoring"):
            anns = img_id_to_anns.get(iid, [])
            s = score_image(
                anns,
                im["width"],
                im["height"],
                args.iou_thr,
                args.contain_thr,
                args.size_frac,
            )
            ranked.append((s, im, anns))
        ranked.sort(key=lambda t: t[0], reverse=True)
        ranked = ranked[: args.count]

    os.makedirs(args.out_dir, exist_ok=True)
    logger.info(f"Loading NuScenes {args.nusc_version} from {args.nusc_root}...")
    nusc = NuScenes(version=args.nusc_version, dataroot=args.nusc_root, verbose=False)
    logger.info("Indexing camera sample_data records by calibrated_sensor token...")
    cs_to_sd = build_cs_to_camera_sd_index(nusc)
    logger.info(f"  {len(cs_to_sd)} unique camera calibrated_sensor tokens.")

    color_map = {}
    csv_path = os.path.join(args.out_dir, "scores.csv")
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["image_id", "score", "file_name", "n_anns", "sd_token"])

        for score, im, anns in tqdm(ranked, desc="rendering pairs"):
            iid = im["id"]
            file_name = im["file_name"]
            cs_token = im.get("other", {}).get("cam_cs_record", {}).get("token")
            coco_img_path = os.path.join(args.img_dir, file_name)
            coco_img_bgr = cv2.imread(coco_img_path)
            if coco_img_bgr is None:
                logger.warning(
                    f"image {iid}: COCO JPG not readable at {coco_img_path}; skipping"
                )
                writer.writerow([iid, f"{score:.2f}", file_name, len(anns), ""])
                continue
            sd_token = match_sd_for_coco_image(nusc, cs_to_sd, coco_img_bgr, cs_token)
            writer.writerow([iid, f"{score:.2f}", file_name, len(anns), sd_token or ""])
            if sd_token is None:
                logger.warning(
                    f"image {iid}: could not pixel-match a sample_data; skipping"
                )
                continue

            out_3d = os.path.join(args.out_dir, f"pair_{iid}_3d.png")
            out_2d = os.path.join(args.out_dir, f"pair_{iid}_2d.png")

            try:
                render_3d(nusc, sd_token, out_3d, show_all=args.show_all_3d)
            except Exception as e:
                logger.error(f"image {iid}: 3D render failed: {e}")
            try:
                render_2d(coco_img_path, anns, cat_id_to_name, out_2d, color_map)
            except Exception as e:
                logger.error(f"image {iid}: 2D render failed: {e}")

    logger.info(f"Done. Wrote {args.count} pairs (and scores.csv) to {args.out_dir}")


if __name__ == "__main__":
    main()
