#!/bin/bash

lr=0.01
steps=10
max_norm=0.01
sigma_0=0.08
init_s=0.08
data=imagenet-sub
root=/nvme0
model=vgg
model_out=./checkpoint/${data}_${model}_adv_vi
echo "model_out: " ${model_out}
CUDA_VISIBLE_DEVICES=4 ./main_adv_vi.py \
                        --lr ${lr} \
                        --step ${steps} \
                        --max_norm ${max_norm} \
                        --sigma_0 ${sigma_0} \
                        --init_s ${init_s} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
                        #--resume \
                        #> >(tee ${model_out}.txt) 2> >(tee error.txt)
