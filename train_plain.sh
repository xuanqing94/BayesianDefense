#!/bin/bash

lr=0.0001
data=cifar10
root=~/data/cifar10-py
model=wresnet
model_out=./checkpoint/${data}_${model}_plain
echo "model_out: " ${model_out}
CUDA_VISIBLE_DEVICES=0,1,2,3 ./main_plain.py \
                        --lr ${lr} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
                        --resume \
                        > >(tee ${model_out}.txt) 2> >(tee error.txt)
