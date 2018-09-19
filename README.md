# BayesianDefense

This is the official repository for paper *Bayesian Defense*

# Data
+ Fashion-MNIST
+ CIFAR10
+ STL10
+ (optional) ImageNet-143 (64px)
+ (optional) ImageNet-1000 (128px)

# Network
+ TinyNet (for Fashion-MNIST), adapted from 
+ VGG16 (for CIFAR10)
+ Aaron (for STL10)
+ ResNet (for ImageNet)

# Defense methods
+ `Plain`: No defense
+ `Adv`: Adversarial training (from )
+ `Adv_vi`: Adversarial training + Variational Inference

# Howto run (for Chongruo)
### Baselines
1. `vim ./train_adv.sh` change `--root` to data path, change `CUDA_VISIBLE_DEVICES` to use as many GPUs as you can.
2. `./train_adv.sh` to execute


### Our method (run it ASAP.)
1. `vim ./train_adv_vi.sh` change `--root` to data path, change `CUDA_VISIBLE_DEVICES` to use ONLY ONE GPU!!!.
2. `./train_adv.sh` to execute


