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
from attacker.pgd import Linf_PGD, L2_PGD
from attacker.cw import cw

model = 'vgg'
defense_vec = ['plain', 'rse', 'adv', 'vi', 'adv_vi']
attack_f = Linf_PGD
data = 'cifar10'
root = '/home/luinx/data/cifar10-py'
eps = 0.06
steps = 20
n_ensemble = 20

# dataset
print('==> Preparing data..')
if data == 'cifar10':
    nclass = 10
    img_width = 32
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=2)
elif data == 'stl10':
    nclass = 10
    img_width = 96
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])
    testset = torchvision.datasets.STL10(root=root, split='test', transform=transform_test, download=True)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=100, shuffle=False)
elif data == 'fashion':
    nclass = 10
    img_width = 28
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.FashionMNIST(root=root, train=False, transform=transform_test, download=True)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=100, shuffle=False)
elif data == 'imagenet-sub':
    nclass = 143
    img_width = 64
    transform_test = transforms.Compose([
        transforms.Resize(img_width),
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.ImageFolder(root+'/sngan_dog_cat_val', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
else:
    raise ValueError(f'invlid dataset: {data}')

def get_model(defense):
    # load model
    if model == 'vgg':
        if defense in ('plain', 'adv'):
            from models.vgg import VGG
            net = nn.DataParallel(VGG('VGG16', nclass, img_width=img_width), device_ids=range(1))
            net.load_state_dict(torch.load(f'./checkpoint/{data}_{model}_{defense}.pth'))
        elif defense in ('vi', 'adv_vi'):
            from models.vgg_vi import VGG
            net = nn.DataParallel(VGG(1.0, 1.0, 1.0, 'VGG16', nclass, img_width=img_width), device_ids=range(1))
        elif defense in ('rse'):
            from models.vgg_rse import VGG
            net = nn.DataParallel(VGG('VGG16', nclass, 0.2, 0.1, img_width=img_width), device_ids=range(1))
    elif model == 'tiny':
        if defense in ('plain', 'adv'):
            from models.tiny import Tiny
            net = nn.DataParallel(Tiny(nclass), device_ids=range(1))
        elif defense in ('vi', 'adv_vi'):
            from models.tiny_vi import Tiny
            net = nn.DataParallel(Tiny(1.0, 1.0, 1.0, nclass), device_ids=range(1))
        elif defense in ('rse'):
            from models.tiny_rse import Tiny
            net = nn.DataParallel(Tiny(nclass, 0.2, 0.1), device_ids=range(1))
    elif model == 'aaron':
        if defense in ('plain', 'adv'):
            from models.aaron import Aaron
            net = nn.DataParallel(Aaron(nclass), device_ids=range(1))
        elif defense in ('vi', 'adv_vi'):
            from models.aaron_vi import Aaron
            net = nn.DataParallel(Aaron(1.0, 1.0, 1.0, nclass), device_ids=range(1))
        elif defense in ('rse'):
            from models.aaron_rse import Aaron
            net = nn.DataParallel(Aaron(nclass, 0.2, 0.1), device_ids=range(1))
    else:
        raise ValueError('invalid model')
    net.load_state_dict(torch.load(f'./checkpoint/{data}_{model}_{defense}.pth'))
    net.cuda()
    net.eval() # must set to evaluation mode
    return net

loss_f = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)
cudnn.benchmark = True

def ensemble_inference(x_in, net, n_ensemble):
    batch = x_in.size(0)
    prob = torch.FloatTensor(batch, nclass).zero_().cuda()
    with torch.no_grad():
        for _ in range(n_ensemble):
            p = softmax(net(x_in)[0])
            prob.add_(p)
    return torch.max(prob, dim=1)[1]

max_iter = 100
defense_vec = defense_vec[::-1]
for i in range(len(defense_vec)):
    for j in range(i+1, len(defense_vec)):
        source_net = get_model(defense_vec[i])
        target_net = get_model(defense_vec[j])
        correct = 0
        total = 0
        for it, (x, y) in enumerate(testloader):
            x, y = x.cuda(), y.cuda()
            x_adv = attack_f(x, y, source_net, steps, eps)
            pred = ensemble_inference(x_adv, target_net, n_ensemble)
            correct += torch.sum(pred.eq(y)).item()
            total += y.numel()
            if it >= max_iter:
                break
        print(f'{defense_vec[i]} ===> {defense_vec[j]}: {correct/total}')
