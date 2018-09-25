#!/bin/bash

model=vgg
defense=adv_vi
data=cifar10

python snr_density.py \
    --model ${model} \
    --defense ${defense} \
    --data ${data}

