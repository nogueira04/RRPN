#!/bin/bash

export PYTHONPATH=~/RRPN
cd ~/RRPN/
source rrpn/bin/activate
cd experiments/
bash 1_generate_proposals.sh
