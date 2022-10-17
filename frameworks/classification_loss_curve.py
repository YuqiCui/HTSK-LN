import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from archs.optims import AdaBound
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import math
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error


def np_rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def label2str(values):
    if not isinstance(values, list):
        values = values.ravel().tolist()
    return "_".join(map(lambda x: "{:.4f}".format(x), values))


def _torch_x2xp(X, frs, n_rules, device):
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    frs = torch.as_tensor(frs, dtype=torch.float32, device=device)
    X = X.unsqueeze(dim=1).expand([X.size(0), n_rules, X.size(1)])  # [n_batch, n_rules, in_dim]
    X = X * frs.unsqueeze(dim=2)
    X = X.view([X.size(0), -1])
    X = torch.cat([X, frs], dim=1)
    return X


def eval_acc_bca(model, loader, device):
    model.eval()
    num_correct = 0
    num_data = 0
    for s, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        out = model(inputs)
        if s == 0:
            out_dim = out.size(1)
            confusion_matrix = torch.zeros(out_dim, out_dim).to(device)

        pred = torch.argmax(out, dim=1)
        num_correct += torch.sum(pred == targets).item()
        num_data += targets.size(0)

        for t, p in zip(targets.view(-1), pred.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    return num_correct / num_data, torch.mean(confusion_matrix.diag() / confusion_matrix.sum(1)).item()


class FrameWork:
    def __init__(self, batch_size=32, val_size=0.1, device="cuda:0", grad_clip=100, batch_print=10, verbose=0):
        self.batch_size = batch_size
        self.val_size = val_size
        self.device = device
        self.grad_clip = grad_clip
        self.batch_print = batch_print
        self.verbose = verbose

    def set_data(self, trainer, valer, tester):
        self.trainer = trainer
        self.valer = valer
        self.tester = tester

    def predict_firing_level(self, loader):
        with torch.no_grad():
            self.model.eval()
            frs = []
            Xs = []
            Ys = []
            for (inputs, targets) in loader:
                inputs = inputs.to(self.device)
                outs = self.model.forward_firing_level(inputs)
                frs.append(outs.detach().cpu().numpy())
                Xs.append(inputs.detach().cpu().numpy())
                Ys.append(targets.detach().cpu().numpy())
        return np.concatenate(frs, axis=0), np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)

    def x2xp(self, X, mem, order=1):
        """
        Converting raw input feature X to TSK consequent input
        :param X: raw input, [n_sample, n_features]
        :param mem: firing level of each rule, [n_sample, n_clusters]
        :param order:order of TSK, 0 or 1
        :return:
        """
        with torch.no_grad():
            return self._torch_x2xp(X, mem).detach().cpu().numpy()

    def _torch_x2xp(self, X, frs):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        frs = torch.as_tensor(frs, dtype=torch.float32, device=self.device)
        X = X.unsqueeze(dim=1).expand([X.size(0), self.model.n_rules, X.size(1)])  # [n_batch, n_rules, in_dim]
        X = X * frs.unsqueeze(dim=2)
        X = X.view([X.size(0), -1])
        X = torch.cat([X, frs], dim=1)
        return X

    def train(self, model, criterion=None, optimizer=None, epochs=200, weight_decay=1e-5, init_lr=0.01, algo_name="", addl_info=None, save_path=None):
        if addl_info is None:
            addl_info = {}
        self.model = model
        self.model.reset_parameters()
        self.model.to(self.device)

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        batch_cnt = 0
        tol_loss = 0
        hists = []
        timestamp = time.time()
        for e in range(epochs):
            if batch_cnt > epochs:
                break
            for inputs, targets in self.trainer:
                batch_cnt += 1
                if batch_cnt > epochs:
                    break
                self.model.train()
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outs = self.model(inputs)
                if isinstance(outs, tuple):
                    outs = outs[0]

                loss = criterion(outs, targets)
                tol_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()

                grad_info = {}
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        with torch.no_grad():
                            grad_info[n] = torch.norm(p.grad, p=2).item()

                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()

                if batch_cnt % self.batch_print == 0:
                    val_acc, val_bca = eval_acc_bca(model, self.valer, self.device)
                    test_acc, test_bca = eval_acc_bca(model, self.tester, self.device)

                    hists.append(
                        {"batch": batch_cnt,
                         "loss": tol_loss / self.batch_print,
                         "val_acc": val_acc,
                         "val_bca": val_bca,
                         "test_acc": test_acc,
                         "test_bca": test_bca,
                         "algo_name": algo_name,
                         **grad_info,
                         **addl_info
                         }
                    )
                    next_timestamp = time.time()
                    time_interval = next_timestamp - timestamp
                    timestamp = next_timestamp
                    if self.verbose > 0:
                        print("Name:{}, Batch: {:5d}, Loss:{:.4f}, Val ACC(BCA): {:.4f}({:.4f}), Test ACC(BCA): {:.4f}({:4f}), LN stats: {:.4f}, Time: {:.4f}s".format(
                            algo_name, batch_cnt, tol_loss / self.batch_print, val_acc, val_bca, test_acc, test_bca, self.model.ln_stat, time_interval
                    ), end=", ")

                        for n, p in model.named_parameters():
                            if p.grad is not None:
                                    print(n, "{:.4f}".format(grad_info[n]), end=", ")
                    tol_loss = 0
                    print()
        print()
        if save_path is not None:
            print("saving...")
            torch.save(self.model.state_dict(), os.path.join(save_path, "{}_rep_{}_data_{}_batch_{}.pkl".format(algo_name, addl_info["n_rep"], addl_info["data_name"], addl_info["batch_size"])))
        return hists
