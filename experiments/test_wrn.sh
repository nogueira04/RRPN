#!/bin/bash

export PYTHONPATH=~/RRPN

cd ~/RRPN/
source rrpn/bin/activate
cd detectron2/
python3 tools/train_net.py --config-file configs/COCO-Detection/faster_rcnn_WideResNet_FPN.yaml --num-gpus 1 OUTPUT_DIR ./output/wide_resnet_faster_rcnn_3

