import torch.nn as nn
from torch.optim import Adam


def get_optim(model, weight_decay=1e-3, init_lr=0.01, center_param_name=None):
    center_params, other_params = [], []
    for n, p in model.named_parameters():
        if n in center_param_name:
            center_params.append(p)
        else:
            other_params.append(p)
    optimizer = Adam(
        [
            {'params': center_params},
            {'params': other_params, 'weight_decay': weight_decay},
        ], lr=init_lr, weight_decay=0
    )
    return optimizer

