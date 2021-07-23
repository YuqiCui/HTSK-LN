import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from archs.optims import AdaBound
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.linear_model import RidgeClassifierCV, RidgeCV, LogisticRegressionCV


def eval_bca(model, loader, device):
    model.eval()
    outs = []
    labels = []
    for s, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        out = model(inputs)
        if isinstance(out, tuple):
            out = out[0]
        # out = F.softmax(out, dim=1)
        outs.append(out.detach().cpu().numpy().argmax(axis=-1))
        labels.append(targets.detach().cpu().numpy())
    outs = np.concatenate(outs, axis=0)
    labels = np.concatenate(labels, axis=0)
    return balanced_accuracy_score(labels, outs)


def eval_acc(model, loader, device):
    model.eval()
    num_correct = 0
    num_data = 0
    for s, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        out = model(inputs)
        if isinstance(out, tuple):
            out = out[0]
        pred = torch.argmax(out, dim=1)
        num_correct += torch.sum(pred == targets).item()
        num_data += targets.size(0)
    return num_correct / num_data


def eval_mse(model, loader, device):
    model.eval()
    tol_error = 0
    num_data = 0
    for s, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        out = model(inputs)
        if isinstance(out, tuple):
            out = out[0]
        tol_error += ((out - targets)**2).sum().item()
        num_data += inputs.size(0)
    return tol_error / num_data


def check_grad(model):
    """
    check if gradients of each model contain nan
    :param model: pytorch model
    :return: bool, True if gradients contain nan, else False
    """
    for n, p in model.named_parameters():
        if torch.sum(torch.isnan(p.grad)) > 0:
            print("Warning: Gradient contains nan, skipping this batch...")
            return True
    return False


class Wrapper:
    def __init__(self,
                 model,
                 device="cuda:0",
                 optimizer_type="adabound",
                 loss_type="mse",
                 init_lr=0.01,
                 weight_decay=0,
                 epochs=200,
                 patience=60,
                 val_metric="mse",
                 val_size=0.1,
                 batch_size=512,
                 grad_clip=None,
                 ur=None,
                 verbose=0,
                 save_path="tmp.pkl",
                 refit_by="svm",
                 class_weight=None,
                 ):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.optimizer_type = optimizer_type
        self.loss_type = loss_type
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.val_metric = val_metric
        self.val_size = val_size
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.verbose = verbose
        self.save_path = save_path
        self.refit_by = refit_by
        self.ur = ur
        self.class_weight = class_weight if class_weight is not None else [1] * self.model.out_dim

    def _get_optims(self):
        center_params = [p for n, p in self.model.named_parameters() if n in ["centers", "sigmas"]]
        other_params = [p for n, p in self.model.named_parameters() if n not in ["centers", "sigmas"]]
        if self.optimizer_type == "adabound":
            # print("Using Adabound as optimizer")
            optimizer = AdaBound(
                [
                    {'params': center_params},
                    {'params': other_params, 'weight_decay': self.weight_decay}
                ], lr=self.init_lr, weight_decay=0
            )
        elif self.optimizer_type == "adam":
            # print("Using Adam as optimizer")
            optimizer = torch.optim.Adam(
                [
                    {'params': center_params},
                    {'params': other_params, 'weight_decay': self.weight_decay}
                ], lr=self.init_lr, weight_decay=0
            )
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                [
                    {'params': center_params},
                    {'params': other_params, 'weight_decay': self.weight_decay}
                ], lr=self.init_lr, weight_decay=0, momentum=0.9
            )
        else:
            raise ValueError("Unsupported optimizer_type")
        return optimizer

    def _init_train_variables(self, X, y):
        optimizer = self._get_optims()

        if self.loss_type == "mse":
            # print("Using MSE as Loss")
            loss_func = nn.MSELoss()
        elif self.loss_type == "ce" or self.loss_type == "crossentropy":
            # print("Using CrossEntropyLoss as Loss")
            loss_func = nn.CrossEntropyLoss(torch.FloatTensor(self.class_weight).to(self.device) if self.class_weight is not None else None)
        else:
            raise ValueError("Unsupported loss_type")

        if self.val_metric == "mse":
            # print("Using MSE as Val metric")
            val_func = eval_mse
            best_val_metric = float("inf")
            is_better = lambda val, best: val < best - 1e-4
        elif self.val_metric == "acc" or self.val_metric == "accuracy":
            # print("Using ACC as Val metric")
            val_func = eval_acc
            best_val_metric = 0
            is_better = lambda val, best: val > best + 1e-4
        elif self.val_metric == "bca":
            # print("Using BCA as Val metric")
            val_func = eval_bca
            best_val_metric = 0
            is_better = lambda val, best: val > best + 1e-4
        else:
            raise ValueError("Unsupported val_metric")
        self.is_better = is_better

        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size)
        self.train_set = [x_train, y_train]
        self.val_set = [x_val, y_val]
        trainer = DataLoader(
            dataset=TensorDataset(
                torch.FloatTensor(x_train),
                torch.FloatTensor(y_train) if self.loss_type == "mse" else torch.LongTensor(y_train)
            ),
            batch_size=min(self.batch_size, x_train.shape[0]),
            shuffle=True,
        )

        valer = DataLoader(
            dataset=TensorDataset(
                torch.FloatTensor(x_val),
                torch.FloatTensor(y_val) if self.loss_type == "mse" else torch.LongTensor(y_val)
            ),
            batch_size=min(self.batch_size, x_val.shape[0]),
            shuffle=False,
        )

        patience_cnt = 0

        return optimizer, loss_func, val_func, best_val_metric, is_better, trainer, valer, patience_cnt

    def set_hook(self, hook):
        self.hook = hook

    def __fit__(self, X, y):
        self.model.reset_parameters()
        optimizer, loss_func, val_func, best_val_metric, is_better, trainer, valer, patience_cnt = self._init_train_variables(X, y)

        for e in range(self.epochs):
            self.model.train()
            tol_loss = 0
            ur_loss = 0

            for inputs, targets in trainer:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                if self.ur is not None:
                    outs, frs = self.model(inputs, with_frs=True)
                    loss_cur = loss_func(outs, targets)
                    ur_loss_cur = self.ur * ((torch.mean(frs, dim=0) - 1 / min(2, self.model.out_dim)) ** 2).sum()

                    tol_loss += loss_cur.item()
                    loss = loss_cur + ur_loss_cur
                    ur_loss += ur_loss_cur.item()
                else:
                    outs = self.model(inputs)
                    if isinstance(outs, tuple):
                        outs = outs[0]
                    loss = loss_func(outs, targets)
                    tol_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                if check_grad(self.model):
                    optimizer.zero_grad()
                    continue
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()

            if self.verbose:
                print("[{:5d}] Loss: {:.4f}, UR Loss: {:.4f}".format(e, tol_loss, ur_loss), end=", ")

            val_metric_value = val_func(self.model, valer, self.device)
            if self.verbose:
                print("Val {}: {:.4f}".format(self.val_metric.upper(), val_metric_value))
            if is_better(val_metric_value, best_val_metric):
                best_val_metric = val_metric_value
                patience_cnt = 0
                torch.save(self.model.state_dict(), self.save_path)
            else:
                patience_cnt += 1

            if patience_cnt > self.patience:
                if self.verbose:
                    print("Early stopping end at {} epoch".format(e))
                break

        self.model.load_state_dict(torch.load(self.save_path))
        self.best_val_metric = best_val_metric
        return self

    def predict_firing_level(self, X, y=None):
        with torch.no_grad():
            n = X.shape[0]
            tester = DataLoader(
                dataset=TensorDataset(
                    torch.FloatTensor(X),
                ),
                batch_size=min(self.batch_size, n),
                shuffle=False,
            )
            self.model.eval()
            frs = []
            for (inputs, ) in tester:
                inputs = inputs.to(self.device)
                outs = self.model.forward_firing_level(inputs)
                frs.append(outs.detach().cpu().numpy())
        return np.concatenate(frs, axis=0)

    def fit(self, X, y=None):
        self.__fit__(X, y)
        frs = self.predict_firing_level(X)
        xp = self.x2xp(X, frs, order=1)
        if self.model.out_dim == 1:  # regression
            refit_svm = make_pipeline(
                GridSearchCV(
                    LinearSVR(), {'C': [0.01, 0.1, 1, 10, 100]},
                    n_jobs=10, cv=5, verbose=0))
            refit_ridge = make_pipeline(
                RidgeCV(alphas=(1e-4, 0.01, 0.1, 1, 10, 100))
            )
            self.refit_models = [
                ("ridge", refit_ridge),
            ]
        else:
            refit_ridge = make_pipeline(
                # StandardScaler(),
                RidgeClassifierCV(alphas=(1e-4, 0.01, 0.1, 1, 10, 100), class_weight={i: self.class_weight[i] for i in range(len(self.class_weight))})
            )
            refit_lr = make_pipeline(
                # StandardScaler(),
                LogisticRegressionCV(Cs=(1e-4, 0.01, 0.1, 1, 10, 100), class_weight={i: self.class_weight[i] for i in range(len(self.class_weight))})
            )
            self.refit_models = [
                ("ridge", refit_ridge),
                # ("LR", refit_lr)
            ]
        for model_name, model in self.refit_models:
            model.fit(xp, y)
        return self

    def predict(self, X, y=None, argmax=True):
        with torch.no_grad():
            n = X.shape[0]
            tester = DataLoader(
                dataset=TensorDataset(
                    torch.FloatTensor(X),
                ),
                batch_size=min(self.batch_size, n),
                shuffle=False,
            )
            self.model.eval()
            y_pred = []
            for (inputs, ) in tester:
                inputs = inputs.to(self.device)
                outs = self.model(inputs)
                if isinstance(outs, tuple):
                    outs = outs[0]
                y_pred.append(outs.detach().cpu().numpy())
            concat = np.concatenate(y_pred, axis=0)
            return concat

    def grad_wrt_input_classification(self, X):
        data_grads = [None] * self.model.out_dim
        n = X.shape[0]
        for i in range(self.model.out_dim):
            input = torch.FloatTensor(X).to(self.device)
            input.requires_grad = True
            output = self.model(input)
            if isinstance(output, tuple):
                output = output[0]
            label = torch.FloatTensor([i] * n).view(-1).to(self.device)
            loss = nn.MSELoss()(output, label)

            self.model.zero_grad()
            loss.backward()
            data_grads[i] = input.grad.detach().cpu().numpy()

        return data_grads

    def predict_refit(self, X, y=None):
        xp = self.x2xp(X, self.predict_firing_level(X), order=1)
        for model_name, model in self.refit_models:
            if model_name == "ridge":
                y_pred = model.decision_function(xp) if self.model.out_dim > 1 else model.predict(xp)
                if self.model.out_dim > 1 and y_pred.ndim == 1:
                    y_pred = np.c_[1 - y_pred, y_pred]
                return y_pred

    def generate_new_cons(self):
        self.newcons = nn.Linear((self.model.in_dim + 1) * self.model.n_rules, 1)
        self.newcons.to(self.device)

        self.newcons.weight.data[...] = torch.as_tensor(self.refit_models[0][1][0].coef_, dtype=torch.float32, device=self.device)
        self.newcons.bias.data[...] = torch.as_tensor(self.refit_models[0][1][0].intercept_, dtype=torch.float32, device=self.device)

        def forward_with_refit(inputs):
            self.model.eval()
            frs = self.model.forward_firing_level(inputs)
            xp = self._torch_x2xp(inputs, frs)
            return self.newcons(xp)
        self.forward_with_refit = forward_with_refit


    def test_refit_metrics(self, X, y, metrics: callable):
        ans_results = []
        xp = self.x2xp(
            X, self.predict_firing_level(X), order=1
        )
        for model_name, model in self.refit_models:
            ans_results.append(
                (model_name, metrics(y, model.predict(xp)))
            )
        return ans_results

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


class ANFISWrapper(Wrapper):
    def __init__(self,
                 model,
                 device="cuda:0",
                 optimizer_type="adabound",
                 loss_type="mse",
                 init_lr=0.01,
                 weight_decay=0,
                 epochs=200,
                 patience=60,
                 val_metric="mse",
                 val_size=0.1,
                 batch_size=512,
                 grad_clip=None,
                 verbose=0,
                 save_path="tmp.pkl",
                 refit_by="svm",
                 class_weight=None,
                 ):
        super(ANFISWrapper, self).__init__(
            model=model,
            device=device,
            optimizer_type=optimizer_type,
            loss_type=loss_type,
            init_lr=init_lr,
            weight_decay=weight_decay,
            epochs=epochs,
            patience=patience,
            val_metric=val_metric,
            val_size=val_size,
            batch_size=batch_size,
            grad_clip=grad_clip,
            verbose=verbose,
            save_path=save_path,
            refit_by=refit_by,
            class_weight=class_weight,
        )

    def _get_optims(self):
        center_params = [p for n, p in self.model.named_parameters() if n in ["centers", "sigmas"]]
        if self.optimizer_type == "adabound":
            # print("Using Adabound as optimizer")
            optimizer = AdaBound(
                [
                    {'params': center_params},
                ], lr=self.init_lr, weight_decay=0
            )
        elif self.optimizer_type == "adam":
            # print("Using Adam as optimizer")
            optimizer = torch.optim.Adam(
                [
                    {'params': center_params},
                ], lr=self.init_lr, weight_decay=0
            )
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                [
                    {'params': center_params},
                ], lr=self.init_lr, weight_decay=0, momentum=0.9
            )
        else:
            raise ValueError("Unsupported optimizer_type")
        return optimizer

    def _torch_x2xp(self, X, frs):
        with torch.no_grad():
            X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            frs = torch.as_tensor(frs, dtype=torch.float32, device=self.device)
            X = X.unsqueeze(dim=1).expand([X.size(0), self.model.n_rules, X.size(1)])  # [n_batch, n_rules, in_dim]
            X = X * frs.unsqueeze(dim=2)
            X = X.view([X.size(0), -1])
            X = torch.cat([X, frs], dim=1)
        return X.detach().cpu().numpy()

    def __fit__(self, X, y=None):
        self.model.reset_parameters()
        optimizer, loss_func, val_func, best_val_metric, is_better, trainer, valer, patience_cnt = self._init_train_variables(X, y)

        for e in range(self.epochs):
            self.model.train()
            tol_loss = 0
            for inputs, targets in trainer:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outs = self.model(inputs)
                if isinstance(outs, tuple):
                    outs = outs[0]

                loss = loss_func(outs, targets)

                optimizer.zero_grad()
                loss.backward()
                if check_grad(self.model):
                    optimizer.zero_grad()
                    continue
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                tol_loss += loss.item()
                optimizer.step()

                firing_levels = self.predict_firing_level(X)
                if self.model.out_dim == 1:
                    ridge = RidgeCV(alphas=(0.01, 0.1, 1, 10, 100))
                else:
                    ridge = RidgeClassifierCV(alphas=(0.01, 0.1, 1, 10, 100))
                x2xp = self._torch_x2xp(X, firing_levels)
                ridge.fit(x2xp, y)
                self.model.cons.weight.data[...] = torch.as_tensor(ridge.coef_, dtype=torch.float32, device=self.device)
                self.model.cons.bias.data[...] = torch.as_tensor(ridge.intercept_, dtype=torch.float32, device=self.device)

            if self.verbose:
                print("[{:5d}] Loss: {:.4f}".format(e, tol_loss), end=", ")

            val_metric_value = val_func(self.model, valer, self.device)
            if self.verbose:
                print("Val {}: {:.4f}".format(self.val_metric.upper(), val_metric_value))
            if is_better(val_metric_value, best_val_metric):
                best_val_metric = val_metric_value
                patience_cnt = 0
                torch.save(self.model.state_dict(), self.save_path)
            else:
                patience_cnt += 1

            if patience_cnt > self.patience:
                if self.verbose:
                    print("Early stopping end at {} epoch".format(e))
                break

        self.model.load_state_dict(torch.load(self.save_path))
        return self



class URWrapper(Wrapper):
    def __init__(self,
                 model,
                 device="cuda:0",
                 optimizer_type="adabound",
                 loss_type="mse",
                 init_lr=0.01,
                 weight_decay=0,
                 epochs=200,
                 patience=60,
                 val_metric="mse",
                 val_size=0.1,
                 batch_size=512,
                 grad_clip=None,
                 ur_list=None,
                 ur_rep=5,
                 verbose=0,
                 save_path="tmp.pkl",
                 refit_by="svm",
                 class_weight=None,
                 ):
        super(URWrapper, self).__init__(
            model=model,
            device=device,
            optimizer_type=optimizer_type,
            loss_type=loss_type,
            init_lr=init_lr,
            weight_decay=weight_decay,
            epochs=epochs,
            patience=patience,
            val_metric=val_metric,
            val_size=val_size,
            batch_size=batch_size,
            grad_clip=grad_clip,
            ur=None,
            verbose=verbose,
            save_path=save_path,
            refit_by=refit_by,
            class_weight=class_weight,
        )
        self.ur_list = ur_list
        self.ur_rep = ur_rep
        self.verbose_bak = self.verbose

    def fit(self, X, y=None):
        assert self.val_metric == "mse" or self.val_metric == "mean_square_error"
        vals_metric = np.zeros([len(self.ur_list), self.ur_rep])
        for idx, ur in enumerate(self.ur_list):
            self.ur = ur
            self.verbose = 0
            # print("Searching UR weight: {}".format(ur))
            for r in tqdm(range(5), bar_format="Searching UR weight: %.1f {l_bar}{bar}{r_bar}"%(ur)):
                self.__fit__(X, y.copy())
                refit_ridge = make_pipeline(
                    RidgeCV(alphas=(1e-4, 0.01, 0.1, 1, 10, 100))
                )
                frs = self.predict_firing_level(X)
                xp = self.x2xp(X, frs, order=1)

                refit_ridge.fit(xp, y)
                vals_metric[idx, r] = np.sqrt(
                    mean_squared_error(self.val_set[1], refit_ridge.predict(self.x2xp(
                        self.val_set[0], self.predict_firing_level(self.val_set[0]), order=1
                    )))
                )
        vals_metric = np.mean(vals_metric, axis=1)
        if self.is_better(1, 2):
            print("argmin")
            idx = np.argmin(vals_metric)
        else:
            idx = np.argmax(vals_metric)
        best_ur = self.ur_list[idx]
        print("select best UR: {}".format(best_ur), vals_metric)
        self.ur = best_ur
        self.verbose = self.verbose_bak
        self.__fit__(X, y)

        frs = self.predict_firing_level(X)
        xp = self.x2xp(X, frs, order=1)
        if self.model.out_dim == 1:  # regression
            refit_ridge = make_pipeline(
                RidgeCV(alphas=(1e-4, 0.01, 0.1, 1, 10, 100))
            )
            self.refit_models = [
                ("ridge", refit_ridge),
            ]
        else:
            refit_ridge = make_pipeline(
                RidgeClassifierCV(alphas=(1e-4, 0.01, 0.1, 1, 10, 100),
                                  class_weight={i: self.class_weight[i] for i in range(len(self.class_weight))})
            )
            self.refit_models = [
                ("ridge", refit_ridge),
            ]
        for model_name, model in self.refit_models:
            model.fit(xp, y)
        return self


