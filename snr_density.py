import argparse
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# arguments
parser = argparse.ArgumentParser(description='Density plot')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--defense', type=str, required=True)
parser.add_argument('--data', type=str, required=True)

opt = parser.parse_args()

if opt.data == 'cifar10':
    nclass = 10
    img_width = 32
elif opt.data == 'stl10':
    nclass = 10
    img_width = 96
elif opt.data == 'imagenet-sub':
    nclass = 143
    img_width = 64
else:
    raise ValueError(f'Invalid dataset {opt.data}')

# load model
if opt.model == 'vgg':
    from models.vgg_vi import VGG
    net = nn.DataParallel(VGG(1.0, 1.0, 1.0, 'VGG16', nclass, img_width=img_width), device_ids=range(1))
elif opt.model == 'tiny':
    from models.tiny_vi import Tiny
    net = nn.DataParallel(Tiny(1.0, 1.0, 1.0, nclass), device_ids=range(1))
elif opt.model == 'aaron':
    from models.aaron_vi import Aaron
    net = nn.DataParallel(Aaron(1.0, 1.0, 1.0, nclass), device_ids=range(1))
else:
    raise ValueError('invalid opt.model')
net.load_state_dict(torch.load(f'./checkpoint/{opt.data}_{opt.model}_{opt.defense}.pth'))


module_set = {'RandConv2d', 'RandBatchNorm2d', 'RandLinear'}
snr_buffer = []

# extract parameters recursively
def extract_param(module):
    modname = type(module).__name__
    if modname in module_set:
        print(modname)
        #snr = (torch.abs(module.mu_weight) / torch.exp(module.sigma_weight)).detach().view(-1).cpu().numpy().tolist()
        snr = module.sigma_weight.detach().view(-1).cpu().numpy().tolist()
        snr_buffer.extend(snr)
    else:
        for submod in module.children():
            extract_param(submod)

extract_param(net)

with open(f'./snr_data/{opt.data}_{opt.model}_{opt.defense}.snr', 'w+') as f:
    f.write('\n'.join([str(s) for s in snr_buffer]))
