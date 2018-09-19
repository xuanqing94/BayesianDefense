#!/bin/bash

model=aaron
defense=adv_vi
data=stl10
root=~/data/stl10
n_ensemble=30
steps=20
max_norm=0,0.03125

echo "Attacking" ./checkpoint/${data}_${model}_${defense}.pth

CUDA_VISIBLE_DEVICES=1 python acc_under_attack.py \
    --model $model \
    --defense $defense \
    --data $data \
    --root $root \
    --n_ensemble $n_ensemble \
    --steps $steps \
    --max_norm $max_norm
