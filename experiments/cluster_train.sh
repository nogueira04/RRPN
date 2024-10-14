#!/bin/bash

export PYTHONPATH=~/RRPN

cd ~/RRPN/
source rrpn/bin/activate
cd experiments/
bash 2_train.sh
