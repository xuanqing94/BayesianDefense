#!/bin/bash

lr=0.01
steps=10
max_norm=0.03125
sigma_0=0.05
init_s=0.05
data=imagenet-sub
root=/data1/xqliu
model=vgg
model_out=./checkpoint/${data}_${model}_adv_vi
echo "model_out: " ${model_out}
CUDA_VISIBLE_DEVICES=0 ./main_adv_vi.py \
                        --lr ${lr} \
                        --step ${steps} \
                        --max_norm ${max_norm} \
                        --sigma_0 ${sigma_0} \
                        --init_s ${init_s} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
                        > >(tee ${model_out}.txt) 2> >(tee error.txt)
