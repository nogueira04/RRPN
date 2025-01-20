#!/bin/bash

export PYTHONPATH=~/RRPN

cd ~/RRPN/
source rrpn/bin/activate
cd detectron2/
python3 tools/train_net.py --config-file configs/COCO-Detection/efficientnet_b7_fpn.yaml --num-gpus 1 OUTPUT_DIR ./output/efficient_net_b7

