import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.noise1 = nn.Parameter(torch.Tensor(out_features, in_features))

        self.register_buffer('eps_weight', torch.Tensor(out_features, in_features))


        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.noise2 = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('eps_bias', torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.noise1.data.zero_()
        self.noise2.data.zero_()

    def forward(self, input):
        weight = self.weight + self.noise1 * self.eps_weight.normal_()
        bias = self.bias + self.noise2 * self.eps_bias.normal_()
        out = F.linear(input, weight, bias)
        return out

layer = MyLinear(10, 1)
layer = nn.DataParallel(layer, device_ids=range(1))
layer.cuda()
x = torch.FloatTensor(20, 10).normal_().cuda()
y = torch.FloatTensor(20).normal_().cuda()
out = layer(x).squeeze()
print(out.size())
diff = out - y
loss = torch.sum(diff * diff) * 0.5
loss.backward()

