#!/bin/bash

model=vgg
defense=adv_vi
data=imagenet-sub
root=/data1/xqliu
n_ensemble=50
steps=( 3 4 6 9 13 18 26 38 55 78 112 162 234 336 483 695 1000 )
attack=Linf
#max_norm=0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07
#max_norm=0,0.002,0.004,0.006,0.008,0.01,0.012,0.014,0.016,0.018,0.02
#max_norm=0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9,1.0
max_norm=0,0.01,0.02
echo "Attacking" ./checkpoint/${data}_${model}_${defense}.pth
for k in "${steps[@]}"
do
    echo "running" $k "steps"
    CUDA_VISIBLE_DEVICES=0 python acc_under_attack.py \
        --model $model \
        --defense $defense \
        --data $data \
        --root $root \
        --n_ensemble $n_ensemble \
        --steps $k \
        --max_norm $max_norm \
        --attack $attack
done
