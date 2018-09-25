import torch
import torch.nn as nn


class Aaron(nn.Module):
    def __init__(self, nclass):
        super(Aaron, self).__init__()
        nchannel = 32
        self.features = nn.Sequential(
                # 3x96x96
                nn.Conv2d(3, nchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(nchannel),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # chx48x48
                nn.Conv2d(nchannel, 2*nchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(2*nchannel),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 2chx24x24
                nn.Conv2d(2*nchannel, 4*nchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(4*nchannel),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 4chx12x12
                nn.Conv2d(4*nchannel, 8*nchannel, kernel_size=3, padding=1),
                nn.BatchNorm2d(8*nchannel),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 8chx6x6
                nn.Conv2d(8*nchannel, 8*nchannel, kernel_size=3, padding=0),
                nn.BatchNorm2d(8*nchannel),
                nn.ReLU(),
                # 8chx4x4
                nn.Conv2d(8*nchannel, 16*nchannel, kernel_size=3, padding=0),
                nn.BatchNorm2d(16*nchannel),
                nn.ReLU(),
                # 8chx2x2
                nn.AvgPool2d(kernel_size=2, stride=2)
                )
        self.classifier = nn.Linear(16*nchannel, nclass)

    def forward(self, input):
        out = self.features(input)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out, None
