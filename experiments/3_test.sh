# Test trained model

# DO NOT EDIT THE NEXT TWO LINES
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"
##------------------------------------------------------------------------------
# Modify these parameters as needed

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${PYTHON:-python3}"
MODEL_PKL="${MODEL_PKL:-$ROOT_DIR/data/models/faster_rcnn_X_101_32x8d_FPN_3x_nucoco_best_trial_2/model_final.pth}"
MODEL_CFG="${MODEL_CFG:-$ROOT_DIR/configs/faster_rcnn_X_101_32x8d_FPN_3x_inf.yaml}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/data/models/faster_rcnn_X_101_32x8d_FPN_3x_nucoco_best_trial_2/val_not_rain_and_night}"
DATASET="${DATASET:-val_not_rain_and_night}"
DATASET_NAME="${DATASET_NAME:-nucoco_$DATASET}"

##------------------------------------------------------------------------------
TEST_PROP_FILES="${TEST_PROP_FILES:-$ROOT_DIR/data/nucoco/proposals/proposals_$DATASET.pkl}"
IMGS_DIR="${IMGS_DIR:-$ROOT_DIR/data/nucoco/$DATASET}"
ANN_FILE="${ANN_FILE:-$ROOT_DIR/data/nucoco/annotations/instances_$DATASET.json}"

echo "INFO: Running inference... "
cd "$ROOT_DIR/det2_port"
"$PYTHON" test_net.py \
    --cfg "$MODEL_CFG" \
    --output-dir "$OUT_DIR" \
    --model-weights "$MODEL_PKL" \
    --proposal-file "$TEST_PROP_FILES" \
    --ann-file "$ANN_FILE" \
    --img-dir "$IMGS_DIR" \
    --dataset-name "$DATASET_NAME" \
    --require-cuda
#    --debug

echo "INFO: Results saved to: $OUT_DIR"
echo "INFO: Done!"
echo "-------------------------------------------------------------------------"
