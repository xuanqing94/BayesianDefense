#!/bin/bash

lr=0.01
sigma_0=0.15
init_s=0.15
data=imagenet-sub
root=/data1/xqliu
model=vgg
model_out=./checkpoint/${data}_${model}_vi
echo "model_out: " ${model_out}
CUDA_VISIBLE_DEVICES=4 python ./main_vi.py \
                        --lr ${lr} \
                        --sigma_0 ${sigma_0} \
                        --init_s ${init_s} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
                        #--resume \
                        #> >(tee ${model_out}.txt) 2> >(tee error.txt)
