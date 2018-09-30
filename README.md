# BayesianDefense

This is the official repository for paper [*Adv-BNN: Improved Adversarial Defense through Robust Bayesian Neural Network*]()

## Data
+ CIFAR10
+ STL10
+ ImageNet-143 (64px)

## Network
+ VGG16 (for CIFAR10/ImageNet-143)
+ Aaron (for STL10)

## Defense methods
+ `Plain`: No defense
+ `RSE`: Random Self-ensemble
+ `Adv`: Adversarial training
+ `Adv_vi`: Adversarial training Bayesian neural network

*Known bugs*: due to a known bug in PyTorch [#11742](https://github.com/pytorch/pytorch/issues/11742), we cannot run RSE/Adv-BNN with multi-GPUs.

## Howto
### Run plain
```bash
lr=0.01
data=imagenet-sub # or `cifar10`, `stl10`
root=/path/to/data
model=vgg # vgg for `cifar10` or `imagenet-sub`, aaron for `stl10`
model_out=./checkpoint/${data}_${model}_plain
echo "model_out: " ${model_out}
CUDA_VISIBLE_DEVICES=3,4 python ./main_plain.py \
                        --lr ${lr} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
```
### Run RSE
```bash
lr=0.01
noise_init=0.2
noise_inner=0.1
data=imagenet-sub # or `cifar10`, `stl10`
root=/path/to/data
model=vgg # vgg for `cifar10` or `imagenet-sub`, aaron for `stl10`
model_out=./checkpoint/${data}_${model}_rse
echo "model_out: " ${model_out}
CUDA_VISIBLE_DEVICES=2 python ./main_rse.py \
                        --lr ${lr} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
                        --noise_init ${noise_init} \
                        --noise_inner ${noise_inner} \
```

### Run Adv. training
```bash
lr=0.01
steps=10
max_norm=0.01
data=imagenet-sub # or `cifar10`, `stl10`
root=/path/to/data
model=vgg # vgg for `cifar10` or `imagenet-sub`, aaron for `stl10`
model_out=./checkpoint/${data}_${model}_adv
echo "model_out: " ${model_out}
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./main_adv.py \
                        --lr ${lr} \
                        --step ${steps} \
                        --max_norm ${max_norm} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
```

### Run our method
```bash
lr=0.01
steps=10
max_norm=0.01
sigma_0=0.1
init_s=0.1
alpha=0.02
data=imagenet-sub
root=/path/to/data
model=vgg
model_out=./checkpoint/${data}_${model}_adv_vi
echo "model_out: " ${model_out}
CUDA_VISIBLE_DEVICES=4 python ./main_adv_vi.py \
                        --lr ${lr} \
                        --step ${steps} \
                        --max_norm ${max_norm} \
                        --sigma_0 ${sigma_0} \
                        --alpha ${alpha} \
                        --init_s ${init_s} \
                        --data ${data} \
                        --model ${model} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
```
