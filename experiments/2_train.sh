# DO NOT EDIT THE NEXT TWO LINES
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"

# Modify these parameters as needed
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${PYTHON:-python3}"

TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-val}"
CFG="${CFG:-$ROOT_DIR/configs/faster_rcnn_X_101_32x8d_FPN_3x.yaml}"
TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$ROOT_DIR/data/models/X-101-32x8d.pkl}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/data/models/faster_rcnn_X_101_32x8d_FPN_3x_rfdetr_base_corrected_base_model}"

# Specify the proposal files
# TRAIN_PROP_FILES="$ROOT_DIR/data/nucoco/proposals/proposals_$TRAIN_SPLIT.pkl"
# TEST_PROP_FILES="$ROOT_DIR/data/nucoco/proposals/proposals_$VAL_SPLIT.pkl"

TRAIN_PROP_FILES="${TRAIN_PROP_FILES:-$ROOT_DIR/data/nucoco/proposals/rfdetr/proposals_${TRAIN_SPLIT}.pkl}"
TEST_PROP_FILES="${TEST_PROP_FILES:-$ROOT_DIR/data/nucoco/proposals/rfdetr/proposals_${VAL_SPLIT}.pkl}"

TRAIN_DATASETS="${TRAIN_DATASETS:-('nucoco_$TRAIN_SPLIT',)}"
TEST_DATASETS="${TEST_DATASETS:-('nucoco_$VAL_SPLIT',)}"
RES_DIR="$OUT_DIR/results"

set -e
mkdir -p "$OUT_DIR"
mkdir -p "$RES_DIR"
cp "$CFG" "$OUT_DIR"

echo "INFO: Starting training..."
cd "$ROOT_DIR/detectron2"
"$PYTHON" tools/train_net.py \
--config-file "$CFG" \
--num-gpus 1 \
OUTPUT_DIR "$OUT_DIR" \
MODEL.WEIGHTS "$TRAIN_WEIGHTS" \
DATASETS.TRAIN "$TRAIN_DATASETS" \
DATASETS.TEST "$TEST_DATASETS" \
DATASETS.PROPOSAL_FILES_TRAIN "('$TRAIN_PROP_FILES',)" \
DATASETS.PROPOSAL_FILES_TEST "('$TEST_PROP_FILES',)"


echo "INFO: Done!"
echo "-------------------------------------------------------------------------"
