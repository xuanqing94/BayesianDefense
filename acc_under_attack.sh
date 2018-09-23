#!/bin/bash

model=vgg
defense=rse
data=imagenet-sub
root=/nvme0
n_ensemble=20
steps=20
max_norm=0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07
#max_norm=0,0.002,0.004,0.006,0.008,0.01,0.012,0.014,0.016,0.018,0.02
echo "Attacking" ./checkpoint/${data}_${model}_${defense}.pth

CUDA_VISIBLE_DEVICES=2 python acc_under_attack.py \
    --model $model \
    --defense $defense \
    --data $data \
    --root $root \
    --n_ensemble $n_ensemble \
    --steps $steps \
    --max_norm $max_norm
