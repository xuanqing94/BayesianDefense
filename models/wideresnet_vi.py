import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.batchnorm2d import RandBatchNorm2d
from .layers.conv2d import RandConv2d
from .layers.linear import RandLinear

class BasicBlock(nn.Module):
    def __init__(self, sigma_0, N, init_s, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = RandBatchNorm2d(sigma_0, N, init_s, in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = RandConv2d(sigma_0, N, init_s, in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = RandBatchNorm2d(sigma_0, N, init_s, out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = RandConv2d(sigma_0, N, init_s, out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and RandConv2d(sigma_0, N, init_s, in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        kl_sum = 0
        tmp, kl = self.bn1(x)
        kl_sum += kl.sum()
        if not self.equalInOut:
            x = self.relu1(tmp)
        else:
            out = self.relu1(tmp)
        tmp, kl = self.conv1(out if self.equalInOut else x)
        kl_sum += kl.sum()
        tmp, kl = self.bn2(tmp)
        kl_sum += kl.sum()
        out = self.relu2(tmp)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out, kl = self.conv2(out)
        kl_sum += kl.sum()
        if self.equalInOut:
            return x + out, kl_sum
        else:
            tmp, kl = self.convShortcut(x)
            kl_sum += kl.sum()
            return tmp + out, kl_sum

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        kl_sum = 0
        for l in self.layer:
            out, kl = l(out)
            kl_sum += kl.sum()
        return out, kl_sum

class WideResNet(nn.Module):
    def __init__(self, sigma_0, N, init_s, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = lambda *args, **kwargs: BasicBlock(sigma_0, N, init_s, *args, **kwargs)
        # 1st conv before any network block
        self.conv1 = RandConv2d(sigma_0, N, init_s, 3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = RandBatchNorm2d(sigma_0, N, init_s, nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = RandLinear(sigma_0, N, init_s, nChannels[3], num_classes)
        self.nChannels = nChannels[3]

    def forward(self, x):
        kl_sum = 0
        out, kl = self.conv1(x)
        kl_sum += kl.sum()
        out, kl = self.block1(out)
        kl_sum += kl
        out, kl = self.block2(out)
        kl_sum += kl
        out, kl = self.block3(out)
        kl_sum += kl
        out, kl = self.bn1(out)
        kl_sum += kl.sum()
        out = self.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out, kl = self.fc(out)
        kl_sum += kl.sum()
        return out, kl_sum


def wresnet(sigma_0, N, init_s, nclass, depth=28, widen_fac=10):
    return WideResNet(sigma_0, N, init_s, depth, nclass, widen_fac, 0)
