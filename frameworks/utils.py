import joblib
import numpy as np
import torch
from collections import defaultdict


def str_1d_numpy(nums):
    s = []
    for num in nums:
        s.append("{:.4f}".format(num))

    return "_".join(s)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def merge_list_of_dicts(L):
    ans = defaultdict(list)
    for ele in L:
        for k, v in ele.items():
            ans[k].append(v)
    return ans


class ConstSplit:
    def __init__(self, path="split.pkl", rep=1):
        self.path = path
        self.rep = rep

    def split(self, X=None, y=None):
        f = joblib.load(self.path)
        train_idxs = f['train_idx']
        test_idxs = f['test_idx']
        N = len(train_idxs)
        for r in range(self.rep):
            for i in range(N):
                yield train_idxs[i], test_idxs[i]


class SpecificSplit:
    def __init__(self, train_idx, test_idx, rep=1):
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.rep = rep

    def split(self, X=None, y=None):
        for r in range(self.rep):
            yield self.train_idx, self.test_idx