# Data-Centric RRPN for Radar-Camera Fusion

This repository supports the accepted VTC 2026 paper **"Data-Centric RRPN:
Boosting Radar-Camera Fusion with High-Fidelity 2D Supervision"**.

The project modernizes the original Radar Region Proposal Network (RRPN) pipeline
from Caffe2/Detectron to PyTorch and Detectron2. The paper keeps the two-stage
RRPN design intact: radar points are projected into the image plane to generate
distance-aware region proposals, then a Fast R-CNN-style detector classifies and
regresses the final boxes. The main contribution is data-centric: RF-DETR is used
to create higher-fidelity 2D supervision for nuScenes images, and the detector is
evaluated with stronger backbones.

## Paper Results

The accepted paper reports COCO-style object-detection metrics on nuScenes-derived
2D detection splits. The metrics below are copied from the paper and are the
canonical values for this release.

### Original 2D Annotations

| Backbone | AP | AP50 | AP75 | AR | ARs | ARm | ARl |
|---|---:|---:|---:|---:|---:|---:|---:|
| X 152 | 31.14 | 59.82 | 29.10 | 25.1 | 16.1 | 34.6 | 50.8 |
| X 101 | 29.83 | 57.18 | 27.59 | 23.9 | 13.7 | 34.2 | 52.6 |
| R 101 | 21.13 | 41.06 | 19.94 | 19.1 | 7.3 | 23.6 | 44.6 |
| W ResNet | 26.17 | 52.11 | 23.62 | 21.8 | 11.2 | 31.6 | 49.3 |
| EffNet B7 | 26.22 | 52.03 | 23.15 | 22.0 | 18.1 | 33.7 | 49.5 |

### RF-DETR 2D Annotations

Superscripts in the paper identify the RF-DETR variant used to generate the
training annotations: `B` for Base and `L` for Large.

| Backbone | AP | AP50 | AP75 | AR | ARs | ARm | ARl |
|---|---:|---:|---:|---:|---:|---:|---:|
| X 152B | 54.21 | 71.75 | 61.06 | 37.8 | 32.9 | 54.3 | 77.1 |
| X 152L | 58.28 | 74.67 | 65.49 | 39.3 | 42.8 | 60.6 | 77.9 |
| X 101B | 28.40 | 35.26 | 30.99 | 21.8 | 4.9 | 23.8 | 47.2 |
| X 101L | 52.91 | 77.39 | 58.46 | 38.5 | 51.0 | 62.3 | 74.9 |
| R 101L | 43.60 | 69.09 | 46.39 | 34.5 | 49.9 | 57.9 | 67.8 |
| EffNet B7L | 39.61 | 65.87 | 41.43 | 31.6 | 45.5 | 53.6 | 63.3 |

### Comparison With Original RRPN

| Backbone | AP | AP50 | AP75 | AR | ARs | ARm | ARl |
|---|---:|---:|---:|---:|---:|---:|---:|
| R 101 (Paper) | 35.5 | 59.0 | 37.0 | 42.1 | 21.1 | 39.9 | 51.4 |
| X 101 (Paper) | 35.4 | 59.2 | 36.9 | 42.0 | 20.2 | 39.0 | 51.0 |
| X 152L (Ours) | 58.3 | 74.7 | 65.5 | 39.3 | 42.8 | 60.6 | 77.9 |
| X 101L (Ours) | 52.9 | 77.4 | 58.5 | 38.5 | 51.0 | 62.3 | 74.9 |
| R 101L (Ours) | 43.6 | 69.1 | 46.4 | 34.5 | 49.9 | 57.9 | 67.8 |

### Per-Class AP

| Backbone | Car | Truck | Person | Motorcycle | Bicycle | Bus |
|---|---:|---:|---:|---:|---:|---:|
| R 101 (Paper) | 41.8 | 44.7 | 17.1 | 30.5 | 21.4 | 57.2 |
| X 101 (Paper) | 41.4 | 44.9 | 17.4 | 29.4 | 21.5 | 57.9 |
| R 101L (Ours) | 57.0 | 50.3 | 41.9 | 35.0 | 24.5 | 52.5 |
| X 101L (Ours) | 61.4 | 58.2 | 52.2 | 47.1 | 35.1 | 63.2 |
| X 152L (Ours) | 72.4 | 69.8 | 63.4 | 57.3 | 49.9 | 68.8 |

## Repository Map

| Path | Purpose |
|---|---|
| `tools/nuscenes_to_coco.py` | Converts nuScenes data to COCO-style 2D annotations using geometric projection. |
| `tools/nuscenes_to_coco_rfdetr.py` | Generates RF-DETR pseudo-label annotations for the data-centric supervision setup. |
| `tools/generate_rrpn_proposals_orig.py` | Generates RRPN proposal pickle files from projected radar points. |
| `det2_port/test_net.py` | Main Detectron2 evaluation entry point using precomputed RRPN proposals. |
| `det2_port/run_conditional_inference.py` | Scene-conditional evaluation entry point for night/rain/other subsets. |
| `configs/` | Detectron2 model configs for the backbones used in the experiments. |
| `experiments/` | Shell wrappers for conversion, proposal generation, training, and evaluation. |
| `detectron2/` | Project-specific Detectron2 fork used by this release. |

Large datasets, proposal pickles, checkpoints, logs, and generated visualizations
are intentionally not committed.

## Data Access

Raw nuScenes data is not redistributed in this repository. Download nuScenes from
the official site and comply with the nuScenes terms of use:

- https://www.nuscenes.org/download
- https://www.nuscenes.org/terms-of-use

The RF-DETR-generated derived annotations and RRPN proposal pickles are released
as a Google Drive artifact:

- Folder: `rrpn-vtc2026-rfdetr-v1`
- Drive folder: https://drive.google.com/drive/folders/1Xy1hnsMNx-3mfNViNG3bVzsFb4rKb-VW
- Archive: `rrpn-vtc2026-rfdetr-v1.tar.gz`
- Archive link: https://drive.google.com/file/d/11YrGgGHv3AhsDYjyVn9f0jjDI9wO64xw/view
- Archive SHA-256:
  `7384b02642c7a7000c3adbc90be90836f571c8f70e6e966bcd454aff03976cd7`
- Access: request access from the repository maintainer if the Drive link is
  restricted.

Expected release layout:

```text
rrpn-vtc2026-rfdetr-v1/
  rfdetr-base/
    annotations/instances_train.json
    annotations/instances_val.json
    proposals/proposals_train.pkl
    proposals/proposals_val.pkl
  rfdetr-large/
    annotations/instances_train.json
    annotations/instances_val.json
    proposals/proposals_train.pkl
    proposals/proposals_val.pkl
  conditional-eval/
    annotations/
      instances_val_night.json
      instances_val_rain.json
      instances_val_not_rain_and_night.json
      instances_val_turn.json
      instances_val_vis20.json
    metadata/
      id_to_scene_val.pkl
  FILELIST.tsv
  MANIFEST.sha256
  DATASET_CARD.md
```

The released dataset artifact excludes raw nuScenes images, copied JPGs,
checkpoints, model weights, TensorBoard logs, and generated visualization folders.
Validate the downloaded archive and extracted package with:

```bash
sha256sum rrpn-vtc2026-rfdetr-v1.tar.gz
tar -xzf rrpn-vtc2026-rfdetr-v1.tar.gz
cd rrpn-vtc2026-rfdetr-v1
sha256sum -c MANIFEST.sha256
```

## Environment

The release code expects a Python 3.9 environment with PyTorch, torchvision,
Detectron2, pycocotools, nuscenes-devkit, OpenCV, PyYAML, Pillow, tqdm, matplotlib,
Optuna, tabulate, RF-DETR, and supervision. The project-specific Detectron2 fork is
included as a submodule and should be installed editable:

```bash
git clone --recurse-submodules https://github.com/nogueira04/RRPN.git
cd RRPN
python -m pip install -r requirements.txt
python -m pip install -e detectron2
```

GPU execution is expected for training and evaluation. The evaluation scripts fail
instead of silently falling back to CPU unless CPU execution is explicitly requested
for debugging.

## Reproduction Workflow

Set the repository root and raw nuScenes location:

```bash
export RRPN_ROOT=$PWD
export NUSC_DIR=/path/to/nuscenes
export CUDA_VISIBLE_DEVICES=0
```

Convert nuScenes to COCO-style annotations:

```bash
bash experiments/0_nuscenes_to_coco.sh --nusc_dir "$NUSC_DIR" --split train
bash experiments/0_nuscenes_to_coco.sh --nusc_dir "$NUSC_DIR" --split val
```

Generate RF-DETR-Large RRPN proposals:

```bash
bash experiments/1_generate_proposals.sh --variant rfdetr_large --split train
bash experiments/1_generate_proposals.sh --variant rfdetr_large --split val
```

Train a model:

```bash
bash experiments/2_train.sh
```

Evaluate a trained model with explicit annotation, image, and proposal paths:

```bash
DATASET=val \
DATASET_NAME=nucoco_val \
TEST_PROP_FILES="$RRPN_ROOT/data/nucoco/proposals/rfdetr_large/proposals_val.pkl" \
ANN_FILE="$RRPN_ROOT/data/nucoco/annotations/rfdetr_large/annotations/instances_val.json" \
IMGS_DIR="$RRPN_ROOT/data/nucoco/annotations/rfdetr_large/val" \
bash experiments/3_test.sh
```

For direct Python evaluation:

```bash
cd det2_port
python test_net.py \
  --cfg "$RRPN_ROOT/configs/faster_rcnn_X_101_32x8d_FPN_3x_inf.yaml" \
  --model-weights "$RRPN_ROOT/data/models/<run>/model_final.pth" \
  --output-dir "$RRPN_ROOT/data/models/<run>/inference" \
  --proposal-file "$RRPN_ROOT/data/nucoco/proposals/rfdetr_large/proposals_val.pkl" \
  --ann-file "$RRPN_ROOT/data/nucoco/annotations/rfdetr_large/annotations/instances_val.json" \
  --img-dir "$RRPN_ROOT/data/nucoco/annotations/rfdetr_large/val" \
  --dataset-name nucoco_val \
  --require-cuda
```

## Expected Outputs

- COCO annotations: `data/nucoco/annotations/.../instances_<split>.json`
- RRPN proposals: `data/nucoco/proposals/<variant>/proposals_<split>.pkl`
- Training run: `data/models/<run>/metrics.json`, `model_final.pth`, and checkpoints
- Evaluation run: `coco_instances_results.json`, `nucoco_val_coco_format.json`,
  `instances_predictions.pth`, visualizations when enabled, and timing plots

## Reproducibility Notes

- The paper uses RF-DETR pseudo-labels as supervision, not as a replacement for
  the RRPN proposal mechanism.
- RF-DETR Base and Large derived datasets are separate artifacts; do not mix their
  annotations or proposal pickles in the same run.
- The released derived dataset includes the `person` class because it is reported
  in the paper's per-class AP table.
- Full training is expensive and writes checkpoints and logs. Use explicit output
  directories and avoid broad `git add` commands.
- Scene-conditional evaluation requires matching split annotations, scene metadata,
  and proposal files.

## Citation

If you use the original RRPN implementation, cite:

```bibtex
@inproceedings{nabati2019rrpn,
  title={RRPN: Radar Region Proposal Network for Object Detection in Autonomous Vehicles},
  author={Nabati, Ramin and Qi, Hairong},
  booktitle={2019 IEEE International Conference on Image Processing (ICIP)},
  pages={3093--3097},
  year={2019},
  organization={IEEE}
}
```
