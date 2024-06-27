# Test trained model

# DO NOT EDIT THE NEXT TWO LINES
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"
##------------------------------------------------------------------------------
# Modify these parameters as needed

export CUDA_VISIBLE_DEVICES=0
MODEL_PKL="$ROOT_DIR/data/models/X_101_32x8d_FPN_1x_nucoco/model_final.pth"
MODEL_CFG="$ROOT_DIR/data/models/faster_rcnn_R_101_FPN_3x.yaml"
OUT_DIR="$ROOT_DIR/data/models/faster_rcnn_R_101_FPN_3x"
DATASET="mini_val"

##------------------------------------------------------------------------------
TEST_PROP_FILES="$ROOT_DIR/data/nucoco/proposals/proposals_$DATASET.pkl"
IMGS_DIR="$ROOT_DIR/data/nucoco/$DATASET"
ANN_FILE="$ROOT_DIR/data/nucoco/annotations/instances_$DATASET.json"

echo "INFO: Running inference... "
cd $ROOT_DIR/det2_port
python3 test_net.py \
    --cfg $MODEL_CFG \
    --vis \
    --output-dir $OUT_DIR \
    TEST.DATASETS "('nucoco_$DATASET',)" \
    TEST.WEIGHTS $MODEL_PKL \
    TEST.PROPOSAL_FILES "('$TEST_PROP_FILES',)" \
    VIS True \

echo "INFO: Results saved to: $OUT_DIR"
echo "INFO: Done!"
echo "-------------------------------------------------------------------------"

