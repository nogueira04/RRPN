#!/bin/bash

cd ~/RRPN/
source rrpn/bin/activate
cd experiments/
bash 0_nuscenes_to_coco.sh
