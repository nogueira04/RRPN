# Generate RRPN proposals from the nucoco dataset

# DO NOT EDIT THE NEXT TWO LINES
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"
##------------------------------------------------------------------------------
# Modify these parameters as needed

SPLIT='train' # 'train' or 'val'

##------------------------------------------------------------------------------
ANN_FILE="$ROOT_DIR/data/nucoco/annotations/instances_${SPLIT}.json"
IMGS_DIR="$ROOT_DIR/data/nucoco/${SPLIT}"
OUT_FILE="$ROOT_DIR/data/nucoco/proposals/proposals_${SPLIT}.pkl"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --split) SPLIT="$2"; shift ;;
        --ann_file) ANN_FILE="$2"; shift ;;
        --imgs_dir) IMGS_DIR="$2"; shift ;;
        --out_file) OUT_FILE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "INFO: Using SPLIT=$SPLIT, ANN_FILE=$ANN_FILE, IMGS_DIR=$IMGS_DIR, OUT_FILE=$OUT_FILE"

echo "INFO: Creating proposals..."

export PYTHONPATH="$ROOT_DIR"
cd $ROOT_DIR/tools
python3 generate_rrpn_proposals.py \
  --ann_file $ANN_FILE \
  --imgs_dir $IMGS_DIR \
  --out_file $OUT_FILE \


echo "INFO: Done!"
echo "-------------------------------------------------------------------------"
