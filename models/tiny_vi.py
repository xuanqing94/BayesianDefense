import torch
import torch.nn as nn
from .layers.batchnorm2d import RandBatchNorm2d
from .layers.conv2d import RandConv2d
from .layers.linear import RandLinear

class Tiny(nn.Module):
    def __init__(self, sigma_0, N, init_s, nclass):
        super(Tiny, self).__init__()
        self.sigma_0 = sigma_0
        self.N = N
        self.init_s = init_s
        self.layers = nn.Sequential(
            RandConv2d(sigma_0, N, init_s, 1, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            RandBatchNorm2d(sigma_0, N, init_s, 32),
            RandConv2d(sigma_0, N, init_s, 32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            RandBatchNorm2d(sigma_0, N, init_s, 64),
            RandConv2d(sigma_0, N, init_s, 64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            RandBatchNorm2d(sigma_0, N, init_s, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            RandConv2d(sigma_0, N, init_s, 128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            RandBatchNorm2d(sigma_0, N, init_s, 256),
            RandConv2d(sigma_0, N, init_s, 256, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            RandBatchNorm2d(sigma_0, N, init_s, 512),
            nn.MaxPool2d(kernel_size=8, stride=1))
        self.fc = RandLinear(sigma_0, N, init_s, 512, nclass)

    def forward(self, x):
        kl_sum = 0
        out = x
        for l in self.layers:
            if type(l).__name__.startswith("Rand"):
                out, kl = l.forward(out)
                if kl is not None:
                    kl_sum += kl
            else:
                out = l.forward(out)
        out = out.view(out.size(0), -1)
        out, kl = self.fc(out)
        kl_sum += kl
        return out, kl_sum

