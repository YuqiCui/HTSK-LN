import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_reg_data(path):
    f = np.load(path)
    return f['data'], f['label']


def load_cls_data(path):
    f = np.load(path)
    # print(f["con_data"].shape, f["label"].shape)
    return f['con_data'], f['label']


def to_float_tensors(*arrays):
    ans = []
    for arr in arrays:
        ans.append(torch.from_numpy(arr).float())
    return ans


def tensor2loader(*args, batch_size=32, shuffle=False):
    train_loader = DataLoader(
        dataset=TensorDataset(*args),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return train_loader