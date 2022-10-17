import os
from collections import defaultdict

import joblib
import torch
import torch.nn as nn
from archs.optims import AdaBound
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import math
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from scipy.stats import entropy


def ur_loss(frs, ur_c):
    return ((torch.mean(frs, dim=0) - 1 / ur_c) ** 2).sum()


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


def eval_rmse(model, loader, device):
    n_data = 0
    n_error = 0
    preds = []
    targets_all = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outs = model(inputs)
            if isinstance(outs, tuple):
                outs = outs[0]
            n_error += torch.sum((outs - targets) ** 2).item()
            n_data += inputs.size(0)

            preds.extend(outs.detach().cpu().numpy().ravel().tolist())
            targets_all.append(targets.detach().cpu().numpy().ravel())
    return math.sqrt(n_error / n_data), preds, np.concatenate(targets_all)


class FrameWork:
    def __init__(self, batch_size=32, val_size=0.1, device="cuda:0", grad_clip=100, batch_print=10, verbose=0):
        self.batch_size = batch_size
        self.val_size = val_size
        self.device = device
        self.grad_clip = grad_clip
        self.batch_print = batch_print
        self.verbose = verbose

    def set_data(self, x_train, y_train, x_test, y_test):
        if y_train.ndim == 1:
            y_train = y_train.reshape([-1, 1])
        if y_test.ndim == 1:
            y_test = y_test.reshape([-1, 1])

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.val_size)
        self.ylabels = [y_train, y_val, y_test]
        self.xdatas = [x_train, x_val, x_test]

        self.trainer = DataLoader(
            dataset=TensorDataset(
                torch.FloatTensor(x_train),
                torch.FloatTensor(y_train),
            ),
            batch_size=min(self.batch_size, x_train.shape[0]),
            shuffle=True,
            drop_last=True,
        )
        self.valer = DataLoader(
            dataset=TensorDataset(
                torch.FloatTensor(x_val),
                torch.FloatTensor(y_val),
            ),
            batch_size=min(self.batch_size, x_train.shape[0]),
            shuffle=False,
        )
        self.tester = DataLoader(
            dataset=TensorDataset(
                torch.FloatTensor(x_test),
                torch.FloatTensor(y_test),
            ),
            batch_size=min(self.batch_size, x_train.shape[0]),
            shuffle=False,
        )

        self.trainer_for_test = DataLoader(
            dataset=TensorDataset(
                torch.FloatTensor(x_train),
                torch.FloatTensor(y_train),
            ),
            batch_size=min(self.batch_size, x_train.shape[0]),
            shuffle=False,
        )

    def train_refit_model(self):
        frs, frs_affine, Xs, Ys = self.predict_firing_level(self.trainer)
        xp = self.x2xp(Xs, frs, order=1)
        self.refit_ridge = RidgeCV(alphas=(0.01, 0.1, 1, 10, 100))
        self.refit_ridge.fit(xp, Ys)

    def predict_refit_model(self, loader):
        frs, frs_affine, Xs, Ys = self.predict_firing_level(loader)
        xp = self.x2xp(Xs, frs, order=1)
        return self.refit_ridge.predict(xp)

    def predict_firing_level(self, loader):
        with torch.no_grad():
            self.model.eval()
            frs = []
            raw_frs = []
            Xs = []
            Ys = []
            for (inputs, targets) in loader:
                inputs = inputs.to(self.device)
                outs = self.model(inputs, with_frs=True)

                frs.append(outs[2].detach().cpu().numpy())
                raw_frs.append(outs[1].detach().cpu().numpy())

                Xs.append(inputs.detach().cpu().numpy())
                Ys.append(targets.detach().cpu().numpy())
        return np.concatenate(raw_frs, axis=0), np.concatenate(frs, axis=0), np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)

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

    def concat_xp_stats(self, loader):
        xp = []
        for inputs, targets in loader:
            inputs = inputs.to(self.device)

            xp.append(
                self.model.output_xp(inputs).detach().cpu().numpy()
            )
        return np.concatenate(xp, axis=0)

    def train(self, model, criterion=None, optimizer=None, epochs=200, weight_decay=1e-5, init_lr=0.01, algo_name="",
              addl_info=None, save_path=None,
              save_path_frs=None, record_xp=False,
              ur_weight=0, ur_c=0.5,
              record_param_change=False):
        if addl_info is None:
            addl_info = {}
        self.model = model
        self.model.reset_parameters()
        self.model.to(self.device)
        self.algo_name = algo_name

        if criterion is None:
            criterion = nn.MSELoss()

        if optimizer is None:
            center_params = [p for n, p in self.model.named_parameters() if n in ["centers", "sigmas"]]
            other_params = [p for n, p in self.model.named_parameters() if n not in ["centers", "sigmas"]]
            optimizer = AdaBound(
                [
                    {'params': center_params},
                    {'params': other_params, 'weight_decay': weight_decay}
                ], lr=init_lr, weight_decay=0
            )

        batch_cnt = 0
        tol_loss = 0
        hists = []
        frs_hists = []
        best_val_rmse, best_test_rmse = float("inf"), float("inf")
        if record_xp:
            self.xp_stats = []

        if record_param_change:
            params_last = {k: v.detach().data.cpu().numpy() for k, v in model.named_parameters()}
            self.params_change = defaultdict(list)

        for e in range(epochs):
            for inputs, targets in self.trainer:
                batch_cnt += 1
                if batch_cnt > epochs:
                    break
                self.model.train()
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outs, raw_frs, frs = self.model(inputs, with_frs=True)

                if save_path_frs is not None:
                    with torch.no_grad():
                        m_raw = torch.mean(raw_frs, dim=0)
                        sd_raw = torch.std(raw_frs, dim=0)
                        m_affine = torch.mean(frs, dim=0)
                        sd_affine = torch.std(frs, dim=0)
                        frs_hists.append([m_raw, sd_raw, m_affine, sd_affine])

                loss = criterion(outs, targets)
                if ur_weight > 0:
                    loss += ur_weight * ur_loss(raw_frs, ur_c)
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
                    val_rmse, val_preds, val_targets = eval_rmse(model, self.valer, self.device)
                    test_rmse, test_preds, test_targets = eval_rmse(model, self.tester, self.device)
                    if val_rmse < best_val_rmse:
                        best_val_rmse = val_rmse
                        best_test_rmse = test_rmse

                    if record_param_change:
                        for k, v in model.named_parameters():
                            self.params_change[k].append(np.linalg.norm(params_last[k] - v.detach().data.cpu().numpy(), ord=2))
                        params_last = {k: v.detach().data.cpu().numpy() for k, v in model.named_parameters()}
                        self.params_change["test_rmse"].append(test_rmse)
                        self.params_change["val_rmse"].append(val_rmse)
                        self.params_change["best_test_rmse"].append(best_test_rmse)

                    raw_frs, frs, _, _ = self.predict_firing_level(self.trainer)
                    ents = entropy(raw_frs, axis=1)
                    m_ents = np.mean(ents).item()
                    std_ents = np.std(ents).item()

                    if record_xp:
                        self.xp_stats.append(
                            (np.mean(raw_frs, axis=0), np.mean(raw_frs, axis=0), test_rmse, val_rmse)
                        )

                    if hasattr(model, "sigmas"):
                        sigs = model.sigmas.detach().cpu().numpy().ravel()
                        sigma_mean = np.mean(sigs).item()
                        sigma_std = np.std(sigs).item()
                    else:
                        sigma_mean = 0
                        sigma_std = 0

                    hists.append(
                        {"batch": batch_cnt,
                         "loss": tol_loss / self.batch_print,
                         "val_rmse": val_rmse,
                         "test_rmse": test_rmse,
                         "best_val_rmse": best_val_rmse,
                         "best_test_rmse": best_test_rmse,
                         "m_ents": m_ents,
                         "std_ents": std_ents,
                         "algo_name": algo_name,
                         "ln_stats": self.model.ln_stat if hasattr(self.model, "ln_stat") else 0,
                         "sigma_mean": sigma_mean,
                         "sigma_std": sigma_std,
                         **grad_info,
                         **addl_info
                         }
                    )

                    if self.verbose > 0:# and batch_cnt == epochs:
                        print_infos = []
                        for metrics in [
                            "algo_name",
                            "batch",
                            "loss",
                            "val_rmse",
                            "best_val_rmse",
                            "test_rmse",
                            "best_test_rmse",
                            # "sigma_mean",
                            # "sigma_std",
                            # "m_ents",
                            # "std_ents",
                            # *list(grad_info.keys())
                        ]:
                            if isinstance(hists[-1][metrics], float):
                                print_infos.append(
                                    "{}: {:.4f}".format(metrics.replace("_", " "), hists[-1][metrics])
                                )
                            else:
                                print_infos.append(
                                    "{}: {}".format(metrics, hists[-1][metrics])
                                )
                        print("{}".format(", ".join(print_infos)))
                    tol_loss = 0
        print()
        if save_path is not None:
            self.actual_save_path = os.path.join(save_path, "{}_rep_{}_data_{}_batch_{}.pkl".format(algo_name, addl_info["n_rep"], addl_info["data_name"], addl_info["batch_size"]))
            torch.save(self.model.state_dict(), self.actual_save_path)

        if save_path_frs is not None:
            joblib.dump(save_path_frs, frs_hists)
        return hists


    def train_others(self, model, criterion=None, optimizer=None, epochs=200, weight_decay=1e-5, init_lr=0.01, algo_name="",
              addl_info=None, save_path=None,
              save_path_frs=None, record_xp=False,
              ur_weight=0, ur_c=0.5,
              record_param_change=False):
        if addl_info is None:
            addl_info = {}
        self.model = model
        self.model.to(self.device)
        self.algo_name = algo_name

        if criterion is None:
            criterion = nn.MSELoss()

        if optimizer is None:
            center_params = [p for n, p in self.model.named_parameters() if n in ["centers", "sigmas"]]
            other_params = [p for n, p in self.model.named_parameters() if n not in ["centers", "sigmas"]]
            optimizer = AdaBound(
                [
                    {'params': center_params},
                    {'params': other_params, 'weight_decay': weight_decay}
                ], lr=init_lr, weight_decay=0
            )

        batch_cnt = 0
        tol_loss = 0
        hists = []
        best_val_rmse, best_test_rmse = float("inf"), float("inf")

        for e in range(epochs):
            for inputs, targets in self.trainer:
                batch_cnt += 1
                if batch_cnt > epochs:
                    break
                self.model.train()
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outs = self.model(inputs)

                loss = criterion(outs, targets)
                tol_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()

                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()

                if batch_cnt % self.batch_print == 0:
                    val_rmse, val_preds, val_targets = eval_rmse(model, self.valer, self.device)
                    test_rmse, test_preds, test_targets = eval_rmse(model, self.tester, self.device)
                    if val_rmse < best_val_rmse:
                        best_val_rmse = val_rmse
                        best_test_rmse = test_rmse

                    hists.append(
                        {"batch": batch_cnt,
                         "loss": tol_loss / self.batch_print,
                         "val_rmse": val_rmse,
                         "test_rmse": test_rmse,
                         "best_val_rmse": best_val_rmse,
                         "best_test_rmse": best_test_rmse,
                         "algo_name": algo_name,
                         "ln_stats": self.model.ln_stat if hasattr(self.model, "ln_stat") else 0,
                         **addl_info
                         }
                    )

                    if self.verbose > 0:# and batch_cnt == epochs:
                        print_infos = []
                        for metrics in [
                            "algo_name",
                            "batch",
                            "loss",
                            "val_rmse",
                            "best_val_rmse",
                            "test_rmse",
                            "best_test_rmse",
                            # "sigma_mean",
                            # "sigma_std",
                            # "m_ents",
                            # "std_ents",
                            # *list(grad_info.keys())
                        ]:
                            if isinstance(hists[-1][metrics], float):
                                print_infos.append(
                                    "{}: {:.4f}".format(metrics.replace("_", " "), hists[-1][metrics])
                                )
                            else:
                                print_infos.append(
                                    "{}: {}".format(metrics, hists[-1][metrics])
                                )
                        print("{}".format(", ".join(print_infos)))
                    tol_loss = 0
        print()
        if save_path is not None:
            self.actual_save_path = os.path.join(save_path, "{}_rep_{}_data_{}_batch_{}.pkl".format(algo_name, addl_info["n_rep"], addl_info["data_name"], addl_info["batch_size"]))
            torch.save(self.model.state_dict(), self.actual_save_path)

        return hists
