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

from models.vgg_noise import VGG
from utils.loss import elbo
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--sigma_0', default=0.05, type=float, help='Prior')
parser.add_argument('--init_s', default=math.log(0.05), type=float)
opt = parser.parse_args()

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='/home/luinx/data/cifar10-py', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/home/luinx/data/cifar10-py', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
net = VGG(opt.sigma_0, len(trainset), opt.init_s, 'VGG16')
net.cuda()
cudnn.benchmark = True

def get_beta(epoch_idx, N):
    return 1.0 / N #* (2 ** (-epoch_idx))

# Training
def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs, kl = net(inputs)
        loss = elbo(outputs, targets, kl, get_beta(epoch, len(trainset)))
        loss.backward()
        optimizer.step()
        pred = torch.max(outputs, dim=1)[1]
        correct += torch.sum(pred.eq(targets)).item()
        total += targets.numel()
    print(f'[TRAIN] Acc: {100.*correct/total:.3f} ({correct}/{total})')


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs, kl = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f'[TEST] Acc: {100.*correct/total:.3f} ({correct}/{total})')
    # Save checkpoint.
    torch.save(net.state_dict(), './checkpoint/vgg16_noise.pth')


epochs = [80, 60, 40, 20]
count = 0

for epoch in epochs:
    optimizer = SGD(net.parameters(), lr=opt.lr, momentum=0.9)
    for _ in range(epoch):
        train(count)
        test(count)
        count += 1
    opt.lr /= 10
