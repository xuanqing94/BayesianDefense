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
parser.add_argument('--model_in', default='./checkpoint/cifar10_vgg_plain.pth', type=str)
parser.add_argument('--eps', type=float, required=True)
opt = parser.parse_args()

# dataset
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='/home/luinx/data/cifar10-py', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=2)

# load model
model = nn.DataParallel(VGG('VGG16', 10))
#model = VGG(1, 1, -1, 'VGG16')
#model = VGG(opt.sigma_0, 1, 'VGG16', init_noise=0.2, inner_noise=0.1, init_s=-1)
#model = ResNet18(opt.sigma_0, 1)
model.load_state_dict(torch.load(opt.model_in))
model.cuda()
model.eval() # must set to evaluation mode
loss_f = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)
cudnn.benchmark = True

def ensemble_inference(x_in, nclass, steps):
    batch = x_in.size(0)
    prob = torch.FloatTensor(batch, nclass).zero_().cuda()
    for _ in range(steps):
        p = softmax(model(x_in)[0])
        prob.add_(p)
    return torch.max(prob, dim=1)[1]

# Iterate over test set
correct = 0
total = 0
max_iter = 100
for it, (x, y) in enumerate(testloader):
    x, y = x.cuda(), y.cuda()
    x_adv = Linf_PGD(x, y, model, 20, opt.eps)
    #pred = torch.max(model(x_adv), dim=1)[1]
    pred = ensemble_inference(x_adv, 10, 20)
    correct += torch.sum(pred.eq(y)).item()
    total += y.numel()
    if it >= max_iter:
        break

print(f'Accuracy: {correct/total}')



