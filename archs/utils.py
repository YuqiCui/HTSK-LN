import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class KeepTopK(nn.Module):
    def __init__(self, k=1, dim=-1, device="cuda:0"):
        self.k = k
        self.dim = dim
        self.device = device
        super(KeepTopK, self).__init__()

    def forward(self, x):
        device_index = x.get_device()
        self.device = "cpu" if device_index == -1 else "cuda:{}".format(device_index)

        topk, indices = torch.topk(x, k=self.k, dim=1, largest=True)
        mask = Variable(torch.zeros_like(x).to(self.device), requires_grad=False).scatter_(self.dim, indices, 1)
        return x * mask


class DropRule(nn.Module):
    def __init__(self, p=0.5):
        self.p = p
        super(DropRule, self).__init__()

    def forward(self, x):
        if self.training:
            binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
            return x * binomial.sample(x.size()).to(x.device)
        return x


def predict_firing_level(model, loader, device):
    with torch.no_grad():
        model.eval()
        frs = []
        raw_frs = []
        Xs = []
        Ys = []
        for (inputs, targets) in loader:
            inputs = inputs.to(device)
            outs = model(inputs, with_frs=True)

            frs.append(outs[2].detach().cpu().numpy())
            raw_frs.append(outs[1].detach().cpu().numpy())

            Xs.append(inputs.detach().cpu().numpy())
            Ys.append(targets.detach().cpu().numpy())
    return np.concatenate(raw_frs, axis=0), np.concatenate(frs, axis=0), np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)

if __name__ == '__main__':
    dr = DropRule(0.5)
    x = torch.ones([2, 32])
    print(dr(x))
