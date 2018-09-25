#!/usr/bin/env python
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torchvision
import torchvision.transforms as transforms

#from models.resnet import ResNet18
from models.vgg import VGG
from attacker.pgd import Linf_PGD

# arguments
parser = argparse.ArgumentParser(description='Bayesian Inference')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--defense', type=str, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--root', type=str, required=True)
parser.add_argument('--n_ensemble', type=int, required=True)
parser.add_argument('--steps', type=int, required=True)
parser.add_argument('--max_norm', type=str, required=True)

opt = parser.parse_args()

opt.max_norm = [float(s) for s in opt.max_norm.split(',')]

# dataset
print('==> Preparing data..')
if opt.data == 'cifar10':
    nclass = 10
    img_width = 32
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root='/home/luinx/data/cifar10-py', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=2)
elif opt.data == 'stl10':
    nclass = 10
    img_width = 96
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])
    testset = torchvision.datasets.STL10(root=opt.root, split='test', transform=transform_test, download=True)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=100, shuffle=False)
elif opt.data == 'fashion':
    nclass = 10
    img_width = 28
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.FashionMNIST(root=opt.root, train=False, transform=transform_test, download=True)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=100, shuffle=False)
elif opt.data == 'imagenet-sub':
    nclass = 143
    img_width = 64
    transform_test = transforms.Compose([
        transforms.Resize(img_width),
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.ImageFolder(opt.root+'/sngan_dog_cat_val', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
else:
    raise ValueError(f'invlid dataset: {opt.data}')

# load model
if opt.model == 'vgg':
    if opt.defense in ('plain', 'adv'):
        from models.vgg import VGG
        net = nn.DataParallel(VGG('VGG16', nclass, img_width=img_width), device_ids=range(1))
        net.load_state_dict(torch.load(f'./checkpoint/{opt.data}_{opt.model}_{opt.defense}.pth'))
    elif opt.defense in ('vi', 'adv_vi'):
        from models.vgg_vi import VGG
        net = nn.DataParallel(VGG(1.0, 1.0, 1.0, 'VGG16', nclass, img_width=img_width), device_ids=range(1))
    elif opt.defense in ('rse'):
        from models.vgg_rse import VGG
        net = nn.DataParallel(VGG('VGG16', nclass, 0.2, 0.1, img_width=img_width), device_ids=range(1))
elif opt.model == 'tiny':
    if opt.defense in ('plain', 'adv'):
        from models.tiny import Tiny
        net = nn.DataParallel(Tiny(nclass), device_ids=range(1))
    elif opt.defense in ('vi', 'adv_vi'):
        from models.tiny_vi import Tiny
        net = nn.DataParallel(Tiny(1.0, 1.0, 1.0, nclass), device_ids=range(1))
    elif opt.defense in ('rse'):
        from models.tiny_rse import Tiny
        net = nn.DataParallel(Tiny(nclass, 0.2, 0.1), device_ids=range(1))
elif opt.model == 'aaron':
    if opt.defense in ('plain', 'adv'):
        from models.aaron import Aaron
        net = nn.DataParallel(Aaron(nclass), device_ids=range(1))
    elif opt.defense in ('vi', 'adv_vi'):
        from models.aaron_vi import Aaron
        net = nn.DataParallel(Aaron(1.0, 1.0, 1.0, nclass), device_ids=range(1))
    elif opt.defense in ('rse'):
        from models.aaron_rse import Aaron
        net = nn.DataParallel(Aaron(nclass, 0.2, 0.1), device_ids=range(1))
else:
    raise ValueError('invalid opt.model')
net.load_state_dict(torch.load(f'./checkpoint/{opt.data}_{opt.model}_{opt.defense}.pth'))
net.cuda()
net.eval() # must set to evaluation mode
loss_f = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)
cudnn.benchmark = True

def ensemble_inference(x_in):
    batch = x_in.size(0)
    prob = torch.FloatTensor(batch, nclass).zero_().cuda()
    with torch.no_grad():
        for _ in range(opt.n_ensemble):
            p = softmax(net(x_in)[0])
            prob.add_(p)
    return torch.max(prob, dim=1)[1]

# Iterate over test set
print('#norm, accuracy')
for eps in opt.max_norm:
    correct = 0
    total = 0
    max_iter = 100
    for it, (x, y) in enumerate(testloader):
        x, y = x.cuda(), y.cuda()
        x_adv = Linf_PGD(x, y, net, opt.steps, eps)
        pred = ensemble_inference(x_adv)
        correct += torch.sum(pred.eq(y)).item()
        total += y.numel()
        if it >= max_iter:
            break
    print(f'{eps}, {correct/total}')



