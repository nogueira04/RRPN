#!/bin/bash

export PYTHONPATH=~/RRPN

cd ~/RRPN/
source rrpn/bin/activate
cd detectron2/
python3 tools/train_net_vit.py --config-file configs/COCO-Detection/timm_vit.yaml --num-gpus 1 OUTPUT_DIR ./output/timm_vit_backbone_2
