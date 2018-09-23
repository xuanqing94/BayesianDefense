#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import Adam, SGD
import torchvision
import torchvision.transforms as transforms

import math
import os
import argparse

from utils.loss import elbo
from attacker.pgd import Linf_PGD

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--steps', default=10, type=int, help='#adv. steps')
parser.add_argument('--max_norm', required=True, type=float, help='Linf-norm in PGD')
parser.add_argument('--sigma_0', required=True, type=float, help='Gaussian prior')
parser.add_argument('--init_s', required=True, type=float, help='Initial log(std) of posterior')
parser.add_argument('--data', required=True, type=str, help='dataset name')
parser.add_argument('--model', required=True, type=str, help='model name')
parser.add_argument('--root', required=True, type=str, help='path to dataset')
parser.add_argument('--model_out', required=True, type=str, help='output path')
parser.add_argument('--resume', action='store_true', help='resume')
opt = parser.parse_args()
opt.init_s = math.log(opt.init_s) # init_s is log(std)
# Data
print('==> Preparing data..')
if opt.data == 'cifar10':
    nclass = 10
    img_width = 32
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root=opt.root, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=opt.root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
elif opt.data == 'stl10':
    nclass = 10
    img_width = 96
    transform_train = transforms.Compose([
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])
    trainset = torchvision.datasets.STL10(root=opt.root, split='train', transform=transform_train, download=True)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True)
    testset = torchvision.datasets.STL10(root=opt.root, split='test', transform=transform_test, download=True)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=100, shuffle=False)

elif opt.data == 'fashion':
    nclass = 10
    img_width = 28
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    trainset = torchvision.datasets.FashionMNIST(root=opt.root, train=True, transform=transform_train, download=True)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True)
    testset = torchvision.datasets.FashionMNIST(root=opt.root, train=False, transform=transform_test, download=True)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=100, shuffle=False)
elif opt.data == 'imagenet-sub':
    nclass = 143
    img_width = 64
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_width, scale=(0.8, 0.9), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(img_width),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.ImageFolder(opt.root+'/sngan_dog_cat', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.ImageFolder(opt.root+'/sngan_dog_cat_val', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
else:
    raise NotImplementedError('Invalid dataset')

# Model
if opt.model == 'vgg':
    from models.vgg_vi import VGG
    net = nn.DataParallel(VGG(opt.sigma_0, len(trainset), opt.init_s, 'VGG16', nclass, img_width=img_width).cuda())
elif opt.model == 'tiny':
    from models.tiny_vi import Tiny
    net = nn.DataParallel(Tiny(opt.sigma_0, len(trainset), opt.init_s, nclass).cuda())
elif opt.model == 'aaron':
    from models.aaron_vi import Aaron
    net = nn.DataParallel(Aaron(opt.sigma_0, len(trainset), opt.init_s, nclass).cuda())
else:
    raise NotImplementedError('Invalid model')

if opt.resume:
    print(f'==> Resuming from {opt.model_out}')
    net.load_state_dict(torch.load(opt.model_out))

cudnn.benchmark = True

def get_beta(epoch_idx, N):
    return 1.0 / N #/ 100

# Training
def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        adv_x = Linf_PGD(inputs, targets, net, opt.steps, opt.max_norm)
        optimizer.zero_grad()
        outputs, kl = net(adv_x)
        loss = elbo(outputs, targets, kl, get_beta(epoch, len(trainset)))
        loss.backward()
        optimizer.step()
        pred = torch.max(outputs, dim=1)[1]
        correct += torch.sum(pred.eq(targets)).item()
        total += targets.numel()
    print(f'[TRAIN] Acc: {100.*correct/total:.3f}')


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, _ = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f'[TEST] Acc: {100.*correct/total:.3f}')
    # Save checkpoint.
    torch.save(net.state_dict(), opt.model_out)


if opt.data == 'cifar10':
    epochs = [80, 60, 40, 20]
elif opt.data == 'imagenet-sub':
    epochs = [30, 20, 20, 10]
elif opt.data == 'fashion':
    epochs = [40, 30, 20]
elif opt.data == 'stl10':
    epochs = [60, 40, 20]
count = 0

for epoch in epochs:
    optimizer = Adam(net.parameters(), lr=opt.lr)
    #optimizer = SGD(net.parameters(), lr=opt.lr, momentum=0.9)
    for _ in range(epoch):
        train(count)
        test(count)
        count += 1
    opt.lr /= 10
