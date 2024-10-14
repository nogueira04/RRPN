# DO NOT EDIT THE NEXT TWO LINES
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"

# Modify these parameters as needed
export CUDA_VISIBLE_DEVICES=0

TRAIN_SPLIT='train'
VAL_SPLIT='val'
CFG="$ROOT_DIR/configs/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
TRAIN_WEIGHTS="$ROOT_DIR/data/models/model_final_68b088.pkl"
OUT_DIR="$ROOT_DIR/data/models/faster_rcnn_X_101_32x8d_FPN_3x_nucoco_WD_02"

# Specify the proposal files
TRAIN_PROP_FILES="$ROOT_DIR/data/nucoco/proposals/proposals_$TRAIN_SPLIT.pkl"
TEST_PROP_FILES="$ROOT_DIR/data/nucoco/proposals/proposals_$VAL_SPLIT.pkl"

TRAIN_DATASETS="('nucoco_$TRAIN_SPLIT',)"
TEST_DATASETS="('nucoco_$VAL_SPLIT',)"
RES_DIR="$OUT_DIR/results"

set -e
mkdir -p $OUT_DIR
mkdir -p $RES_DIR
cp $CFG $OUT_DIR

echo "INFO: Starting training..."
cd $ROOT_DIR/detectron2
python3 tools/train_net.py \
--config-file $CFG \
--num-gpus 1 \
OUTPUT_DIR $OUT_DIR \
MODEL.WEIGHTS $TRAIN_WEIGHTS \
DATASETS.TRAIN $TRAIN_DATASETS \
DATASETS.TEST $TEST_DATASETS \


echo "INFO: Done!"
echo "-------------------------------------------------------------------------"
