#!/bin/bash

lr=0.01
steps=10
max_norm=0.03125
data=cifar10
root=~/data/cifar10-py
model=wresnet
model_out=./checkpoint/${data}_${model}_adv
echo "model_out: " ${model_out}
CUDA_VISIBLE_DEVICES=2,3 ./main_adv.py \
                        --lr ${lr} \
                        --step ${steps} \
                        --max_norm ${max_norm} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
                        > >(tee ${model_out}.txt) 2> >(tee error.txt)
