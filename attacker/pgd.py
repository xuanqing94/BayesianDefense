import torch
import torch.nn.functional as F
from .linf_sgd import Linf_SGD


# performs Linf-constraint PGD attack w/o noise
# @epsilon: radius of Linf-norm ball
def Linf_PGD(x_in, y_true, net, steps, eps):
    training = net.training
    if training:
        net.eval()
    x_adv = x_in.clone().requires_grad_()
    optimizer = Linf_SGD([x_adv], lr=0.007)
    for _ in range(steps):
        optimizer.zero_grad()
        net.zero_grad()
        out, _ = net(x_adv)
        loss = -F.cross_entropy(out, y_true)
        loss.backward()
        optimizer.step()
        diff = x_adv - x_in
        diff.clamp_(-eps, eps)
        x_adv.detach().copy_((diff + x_in).clamp_(0, 1))
    net.zero_grad()
    # reset to the original state
    if training:
        net.train()
    return x_adv

