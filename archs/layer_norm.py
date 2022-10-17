import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNormImpl(nn.Module):
    __constants__ = ['weight', 'bias', 'eps']

    def __init__(self, hidden, mode="adanorm", sigma=1, adanorm_scale=1, nowb_scale=1, mean_detach=False, std_detach=False, eps=1e-5, elementwise_affine=True):
        super(LayerNormImpl, self).__init__()
        self.mode = mode
        self.sigma = sigma
        self.hidden = hidden
        self.adanorm_scale = adanorm_scale
        self.nowb_scale = nowb_scale
        self.mean_detach = mean_detach
        self.std_detach = std_detach
        if self.mode == 'no_norm':
            elementwise_affine = False
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(hidden))
            self.bias = nn.Parameter(torch.Tensor(hidden))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def __repr__(self):
        return "CustomLN(mode={}, affine={})".format(self.mode, self.elementwise_affine)

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if self.mode == 'no_norm':
            if self.elementwise_affine:
                return self.weight * input + self.bias
            else:
                return input
            # return input
        elif self.mode == 'topk':
            T, B, C = input.size()
            input = input.reshape(T * B, C)
            k = max(int(self.hidden * self.sigma), 1)
            input = input.view(1, -1, self.hidden)
            topk_value, topk_index = input.topk(k, dim=-1)
            topk_min_value, top_min_index = input.topk(k, dim=-1, largest=False)
            top_value = topk_value[:, :, -1:]
            top_min_value = topk_min_value[:, :, -1:]
            d0 = torch.arange(top_value.shape[0], dtype=torch.int64)[:, None, None]
            d1 = torch.arange(top_value.shape[1], dtype=torch.int64)[None, :, None]
            input[d0, d1, topk_index] = top_value
            input[d0, d1, top_min_index] = top_min_value
            input = input.reshape(T, B, self.hidden)
            return F.layer_norm(
                input, torch.Size([self.hidden]), self.weight, self.bias, self.eps)
        elif self.mode == 'adanorm':
            mean = input.mean(-1, keepdim=True)
            std = input.std(-1, keepdim=True)
            input = input - mean
            mean = input.mean(-1, keepdim=True)
            graNorm = (1 / 10 * (input - mean) / (std + self.eps)).detach()
            input_norm = (input - input * graNorm) / (std + self.eps)
            return input_norm * self.adanorm_scale
        elif self.mode == 'nowb':
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            if self.mean_detach:
                mean = mean.detach()
            if self.std_detach:
                std = std.detach()
            input_norm = (input - mean) / (std + self.eps)
            return input_norm
        elif self.mode == 'gradnorm':
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            input_norm = (input - mean) / (std + self.eps)
            output = input.detach() + input_norm
            return output
        elif self.mode == "mean":
            mean = input.mean(dim=-1, keepdim=True)
            input_norm = (input - mean)
            if self.elementwise_affine:
                return self.weight * input_norm + self.bias
            else:
                return input_norm
        elif self.mode == "std":
            std = input.std(dim=-1, keepdim=True)
            input_norm = input / (std + self.eps)
            if self.elementwise_affine:
                return self.weight * input_norm + self.bias
            else:
                return input_norm


class LayerNormCut(nn.Module):
    def __init__(self, in_dim, keep="mean", affine=True):
        self.in_dim = in_dim
        self.keep = keep
        self.affine = affine
        super(LayerNormCut, self).__init__()
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(in_dim))
            self.bias = nn.Parameter(torch.Tensor(in_dim))
        self.eps = 1e-5

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if self.keep == "mean":
            mean = input.mean(dim=-1, keepdim=True)
            input_norm = (input - mean)
        elif self.keep == "std":
            std = input.std(dim=-1, keepdim=True)
            input_norm = input / (std + self.eps)
        else:
            raise ValueError("Wrong type of keep, only support mean and std, got {}".format(self.keep))
        if self.affine:
            return self.weight * input_norm + self.bias
        else:
            return input_norm
