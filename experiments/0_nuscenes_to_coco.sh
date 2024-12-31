# Convert the Nuscenes dataset to COCO format

# DO NOT EDIT THE NEXT TWO LINES
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$CUR_DIR")"
##------------------------------------------------------------------------------
## Modify these parameters as needed

NUSC_SPLIT='val'
NUM_RADAR_SWEEPS=1       # number of Radar sweeps
USE_SYMLINKS='False'      # use symlinks instead of copying nuScenes images

##------------------------------------------------------------------------------

NUSC_DIR="/clusterlivenfs/shared_datasets/nuscenes"
OUT_DIR="$ROOT_DIR/data/nucoco"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --nusc_dir) NUSC_DIR="$2"; shift ;;
        --out_dir) OUT_DIR="$2"; shift ;;
        --split) NUSC_SPLIT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "INFO: Using NUSC_DIR=$NUSC_DIR, OUT_DIR=$OUT_DIR, NUSC_SPLIT=$NUSC_SPLIT"

# create symbolic link to the nucoco dataset for Detectron
ln -s $ROOT_DIR/data/nucoco $ROOT_DIR/detectron/detectron/datasets/data/nucoco

echo "INFO: Converting nuScenes to COCO format..."

cd $ROOT_DIR/tools
python3 nuscenes_to_coco.py \
  --nusc_root $NUSC_DIR \
  --split $NUSC_SPLIT \
  --out_dir $OUT_DIR \
  --nsweeps_radar $NUM_RADAR_SWEEPS \
  --use_symlinks False


echo "INFO: Done!"
echo "-------------------------------------------------------------------------"
