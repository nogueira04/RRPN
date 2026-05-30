# Generate RRPN proposals from the nucoco dataset

# DO NOT EDIT THE NEXT TWO LINES
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"
##------------------------------------------------------------------------------
# Modify these parameters as needed

SPLIT="${SPLIT:-train}" # 'train' or 'val'
VARIANT="${VARIANT:-rfdetr_large}" # 'original', 'rfdetr', or 'rfdetr_large'
PYTHON="${PYTHON:-python3}"

##------------------------------------------------------------------------------
# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --split) SPLIT="$2"; shift ;;
        --variant) VARIANT="$2"; shift ;;
        --ann_file) ANN_FILE="$2"; shift ;;
        --imgs_dir) IMGS_DIR="$2"; shift ;;
        --out_file) OUT_FILE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

case "$VARIANT" in
    original)
        DEFAULT_ANN_FILE="$ROOT_DIR/data/nucoco/annotations/instances_${SPLIT}.json"
        DEFAULT_IMGS_DIR="$ROOT_DIR/data/nucoco/${SPLIT}"
        DEFAULT_OUT_FILE="$ROOT_DIR/data/nucoco/proposals/proposals_${SPLIT}.pkl"
        ;;
    rfdetr)
        DEFAULT_ANN_FILE="$ROOT_DIR/data/nucoco/annotations/rfdetr/output_ann/annotations/instances_${SPLIT}.json"
        DEFAULT_IMGS_DIR="$ROOT_DIR/data/nucoco/annotations/rfdetr/output_ann/${SPLIT}"
        if [[ "$SPLIT" == "train" ]]; then
            DEFAULT_ANN_FILE="$ROOT_DIR/data/nucoco/annotations/rfdetr/output_ann_train/annotations/instances_train.json"
            DEFAULT_IMGS_DIR="$ROOT_DIR/data/nucoco/annotations/rfdetr/output_ann_train/train"
        fi
        DEFAULT_OUT_FILE="$ROOT_DIR/data/nucoco/proposals/rfdetr/proposals_${SPLIT}.pkl"
        ;;
    rfdetr_large)
        DEFAULT_ANN_FILE="$ROOT_DIR/data/nucoco/annotations/rfdetr_large/annotations/instances_${SPLIT}.json"
        DEFAULT_IMGS_DIR="$ROOT_DIR/data/nucoco/annotations/rfdetr_large/${SPLIT}"
        DEFAULT_OUT_FILE="$ROOT_DIR/data/nucoco/proposals/rfdetr_large/proposals_${SPLIT}.pkl"
        ;;
    *)
        echo "Unknown variant: $VARIANT"
        exit 1
        ;;
esac

ANN_FILE="${ANN_FILE:-$DEFAULT_ANN_FILE}"
IMGS_DIR="${IMGS_DIR:-$DEFAULT_IMGS_DIR}"
OUT_FILE="${OUT_FILE:-$DEFAULT_OUT_FILE}"

echo "INFO: Using SPLIT=$SPLIT, VARIANT=$VARIANT, ANN_FILE=$ANN_FILE, IMGS_DIR=$IMGS_DIR, OUT_FILE=$OUT_FILE"

echo "INFO: Creating proposals..."

export PYTHONPATH="$ROOT_DIR"
cd "$ROOT_DIR/tools"
"$PYTHON" generate_rrpn_proposals_orig.py \
  --ann_file "$ANN_FILE" \
  --imgs_dir "$IMGS_DIR" \
  --out_file "$OUT_FILE"

# python generate_rrpn_proposals.py \
#     --ann_file_base /clusterlivenfs/gnmp/RRPN/data/nucoco/annotations/instances_val \
#     --imgs_dir_base /clusterlivenfs/gnmp/RRPN/data/nucoco/val \
#     --output_file_base /clusterlivenfs/gnmp/RRPN/data/nucoco/proposals/proposals_val \
#     --keyword_splits night rain not_rain_and_night \
#     --nms_threshold 0.8 \
#     --include_depth 0

echo "INFO: Done!"
echo "-------------------------------------------------------------------------"
