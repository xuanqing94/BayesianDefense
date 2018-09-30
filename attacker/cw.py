import torch
from torch.optim import Adam

n_class = 10
def cw(x_in, y_true, net, steps, eps):
    if eps == 0:
        return x_in
    training = net.training
    if training:
        net.eval()
    index = y_true.cpu().view(-1, 1)
    label_onehot = torch.FloatTensor(x_in.size(0), n_class).zero_().scatter_(1, index, 1).cuda()
    x_adv = x_in.clone().zero_().requires_grad_()
    optimizer = Adam([x_adv], lr=1.0e-2)
    zero = torch.FloatTensor([0]).cuda()
    print('===============')
    for _ in range(steps):
        net.zero_grad()
        optimizer.zero_grad()
        diff = x_adv - x_in
        output, _ = net(x_adv)
        real = torch.max(torch.mul(output, label_onehot), 1)[0]
        other = torch.max(torch.mul(output, (1-label_onehot))-label_onehot*10000, 1)[0]
        error = torch.sum(diff * diff)
        error += eps * torch.sum(torch.max(real - other, zero))
        print(f"error: {error.item()}")
        error.backward()
        optimizer.step()
    return x_adv
