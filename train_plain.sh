#!/bin/bash

lr=0.01
data=stl10
root=~/data/stl10
model=aaron
model_out=./checkpoint/${data}_${model}_plain
echo "model_out: " ${model_out}
CUDA_VISIBLE_DEVICES=1 ./main_plain.py \
                        --lr ${lr} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
                        #> >(tee ${model_out}.txt) 2> >(tee error.txt)
