import math
import os
import torch
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, pairwise_distances
from sklearn.model_selection import RepeatedKFold, ShuffleSplit
from frameworks.utils import ConstSplit, merge_list_of_dicts, str_1d_numpy
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader


def person_cc(y_true, y_pred):
    return np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1]


def tensor2loader(*args, batch_size=32, shuffle=False):
    train_loader = DataLoader(
        dataset=TensorDataset(*args),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return train_loader


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


def get_fft_energy(model, h, nb_interpolation, X, max_num=300):
    if isinstance(model, torch.nn.Module):
        model.eval()
    h = h
    steps = np.arange(-h, h, 2 * h / nb_interpolation)
    with torch.no_grad():
        nb_x = 0
        origin_inputs = X[:max_num]
        nb_x += len(origin_inputs)
        direction = np.random.standard_normal(origin_inputs.shape)
        normed_direction = direction.reshape(len(origin_inputs), -1)
        normed_direction /= np.linalg.norm(normed_direction, ord=2, axis=-1, keepdims=True)
        normed_direction = np.reshape(normed_direction, newshape=origin_inputs.shape)
        results = []
        for step in steps:
            pert_inputs = step * normed_direction + origin_inputs
            result = model.predict(pert_inputs).ravel()
            results.append(result)

        results = np.array(results).transpose(1, 0)
        As = np.abs(np.fft.rfft(results, axis=-1))
    return As


class RegFrameWork:
    def __init__(self, name="FrameWork", verbose=0, save_path=None, scale_y=False):
        self.name = name
        self.verbose = verbose
        self.save_path = save_path
        self.scale_y = scale_y

    def save(self, algo_name="test"):
        ffts = self.run_info["fft"]
        sum_fft = []
        n_fft = []
        for fft in ffts:
            sum_fft.append(str_1d_numpy(np.sum(fft ** 2, axis=0)))
            n_fft.append(fft.shape[0])

        self.run_info['sum_fft'] = sum_fft
        self.run_info["n_fft"] = n_fft
        metrics = ["idx", "rmse", "mae", "cc", "sum_fft", "n_fft", "ridge_rmse", "ridge_mae", "ridge_cc"]
        df = pd.DataFrame(data={k: v for k, v in self.run_info.items() if k in metrics})
        df["name"] = algo_name
        df["data"] = self.name
        if os.path.exists(self.save_path):
            write_head = False
        else:
            write_head = True
        df.to_csv(self.save_path, mode="a", index=False, header=write_head)

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def set_data(self, data, label, sample_weight=None):
        self.data = data
        self.label = label.reshape([-1, 1])
        self.sample_weight = sample_weight
        self.data_info = {
            "in_dim": data.shape[1],
        }
        if self.scale_y:
            self.ymean = label.mean()
            self.ystd = (label - self.ymean).std()

    def set_split_func(self, mode="cv", cv=5, split=0.3, sp_path=None, rep=1):
        self.sp_path = sp_path
        if mode == "cv":
            split_func = RepeatedKFold(n_splits=cv, n_repeats=rep)
        elif mode == "split":
            split_func = ShuffleSplit(n_splits=rep, test_size=split)
        elif mode == "const":
            split_func = ConstSplit(path=sp_path, rep=rep)
        else:
            split_func = mode
        self.split_func = split_func

    def run(self, save_path=None):
        assert hasattr(self, "algorithm"), "Must run set_algorithm() first!"
        assert hasattr(self, "data") and hasattr(self, "label"), "Must run set_data() first!"

        self.run_info = self._rep_fit_predict(self.split_func)
        if save_path is not None:
            assert save_path != self.sp_path, "Save to original const split path, it will cause a loop!"
            joblib.dump(self.run_info, save_path)

    def plot_figure(self):
        assert hasattr(self, "run_info"), "Must run run() first!"
        pds = self.run_info["y_pred"]
        gts = self.run_info["y_test"]
        N = 1
        nrow = int(math.sqrt(N))
        ncol = N // nrow + 1
        print(nrow, ncol, N)
        for i in range(N):
            plt.subplot(nrow, ncol, i + 1)
            plt.plot(gts[i], label="GT")
            plt.plot(pds[i], label="PD")
            if i == N - 1:
                plt.legend()
        plt.show()

    def print_results(self, metrics=None):
        if metrics is None:
            metrics = ["mse", "rmse", "mae", "cc"]
        print("Total Run: {}".format(self.run_info["idx"]))
        for m in metrics:
            print("{}: {:.4f}±{:.4f}".format(
                m.upper(), np.mean(self.run_info[m]), np.std(self.run_info[m]))
            )

    def _fit_predict(self, x_train, y_train, x_test, y_test):
        ss = StandardScaler()
        x_train = ss.fit_transform(x_train)
        x_test = ss.transform(x_test)
        if self.scale_y:
            y_train = (y_train - self.ymean) / self.ystd

        self.algorithm.fit(x_train, y_train)
        y_pred = self.algorithm.predict(x_test)
        if self.scale_y:
            y_pred = y_pred * self.ystd + self.ymean
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        cc = person_cc(y_test, y_pred)

        refit_ans = {}
        try:
            for metname, met in [("rmse", mean_squared_error), ("mse", mean_squared_error), ("mae", mean_absolute_error), ("cc", person_cc)]:
                refit_results = self.algorithm.test_refit_metrics(x_test, y_test, met)
                for n, m in refit_results:
                    if metname == "rmse":
                        m = math.sqrt(m)
                    refit_ans["{}_{}".format(n, metname)] = m
        except:
            for m in ["ridge_rmse", "ridge_mae", "ridge_cc", "ridge_mse"]:
                refit_ans[m] = 0

        fft = get_fft_energy(self.algorithm, 0.1, 300, x_train, max_num=min(x_train.shape[0], 300))
        return {"y_pred": y_pred, "y_test": y_test, "mse": mse, "mae": mae, "cc": cc, "rmse": math.sqrt(mse), "fft": fft, **refit_ans}

    def _rep_fit_predict(self, split_func):
        results = []
        for idx, (train_idx, test_idx) in tqdm(enumerate(split_func.split(self.data, self.label))):
            if self.verbose:
                print("[RegFrameWork IDX: {:5d}]".format(idx))
            x_train, x_test = self.data[train_idx, :], self.data[test_idx, :]
            y_train, y_test = self.label[train_idx], self.label[test_idx]

            ans = dict(
                {"idx": idx, "train_idx": train_idx, "test_idx": test_idx},
                **self._fit_predict(x_train, y_train, x_test, y_test)
            )

            while ans["rmse"] > 1e5:
                ans = dict(
                    {"idx": idx, "train_idx": train_idx, "test_idx": test_idx},
                    **self._fit_predict(x_train, y_train, x_test, y_test)
                )
            results.append(ans)
            if self.verbose:
                print("CurIDX RMSE: {:.4f}, CC: {:.4f}".format(ans["rmse"], ans["cc"]))
        return merge_list_of_dicts(results)


if __name__ == '__main__':
    from sklearn.linear_model import RidgeCV
    x = np.random.rand(1000, 20)
    y = np.random.rand(1000)
    algo = RidgeCV()
    fw = RegFrameWork()
    fw.set_algorithm(algo)
    fw.set_data(x, y)
    fw.run(mode="split", sp_path="test.pkl", rep=10, save_path="test1.pkl")
    fw.plot_figure()

