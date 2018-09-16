#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import Adam
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from attacker.pgd import Linf_PGD

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', required=True, type=float, help='learning rate')
parser.add_argument('--data', required=True, type=str, help='dataset name')
parser.add_argument('--model', required=True, type=str, help='model name')
parser.add_argument('--root', required=True, type=str, help='path to dataset')
parser.add_argument('--model_out', required=True, type=str, help='output path')
parser.add_argument('--resume', action='store_true', help='Resume training')
opt = parser.parse_args()

# Data
print('==> Preparing data..')
if opt.data == 'cifar10':
    nclass = 10
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
elif opt.data == 'imagenet-sub':
    raise NotImplementedError
else:
    raise NotImplementedError('Invalid dataset')

# Model
if opt.model == 'vgg':
    from models.vgg import VGG
    net = nn.DataParallel(VGG('VGG16').cuda())
elif opt.model == 'wresnet':
    from models.wideresnet import wresnet
    net = nn.DataParallel(wresnet(nclass, 28, 10).cuda())
else:
    raise NotImplementedError('Invalid model')

if opt.resume:
    print(f'==> Resume from {opt.model_out}')
    net.load_state_dict(torch.load(opt.model_out))

cudnn.benchmark = True

# Loss function
criterion = nn.CrossEntropyLoss()

# Training
def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        # No adv. training
        #x_adv = Linf_PGD(inputs, targets, net, 10, 0.03125)
        optimizer.zero_grad()
        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)
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
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f'[TEST] Acc: {100.*correct/total:.3f}')

    # Save checkpoint after each epoch
    torch.save(net.state_dict(), opt.model_out)

# Go
epochs = [80, 60, 40, 20]
count = 0
for epoch in epochs:
    optimizer = Adam(net.parameters(), lr=opt.lr)
    for _ in range(epoch):
        train(count)
        test(count)
        count += 1
    opt.lr /= 10
