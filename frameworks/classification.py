import math

import numpy as np
import torch
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, pairwise_distances
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from frameworks.utils import merge_list_of_dicts, ConstSplit, str_1d_numpy
import joblib
from tqdm import tqdm


def get_fft_energy(model, h, nb_interpolation, X, max_num=300):
    max_num = min(max_num, X.shape[0])
    pdist = pairwise_distances(X)
    pdist = np.where(pdist < 1e-4, 1e20, pdist)
    h = np.min(pdist) * h
    # print(h, pairwise_distances(X)[np.triu_indices(X.shape[0], 1)])
    prob_list = []
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
            # result = torch.nn.Softmax(-1)(model(pert_inputs)).data.cpu().numpy()
            result = model.predict(pert_inputs)
            results.append(result)
        results = np.array(results).transpose(1, 2, 0)
        A = np.abs(np.fft.rfft(results, axis=-1))
        prob_list.append(A)

        As = np.concatenate(prob_list, axis=0)
    # print(As.shape)
    return As


class ClsFrameWork:
    def __init__(self, name="Framework", verbose=0, save_path=None):
        self.name = name
        self.verbose = verbose
        self.save_path = save_path

    def save(self, algo_name="test"):
        # fft = sum(self.run_info["fft"])
        # sum_fft = sum(self.run_info["sum_fft"])
        # print("save")
        if self.compute_fft:
            self.run_info["fft"] = [str_1d_numpy(x) for x in self.run_info['fft']]

        metrics = ["idx", "acc", "bca", "fft", "sum_fft", "ridge_acc", "ridge_bca", "LR_acc", "LR_bca"]
        df = pd.DataFrame(data={k: v for k, v in self.run_info.items() if k in metrics})
        df["name"] = algo_name
        df["data"] = self.name
        if os.path.exists(self.save_path):
            write_head = False
        else:
            write_head = True
        df.to_csv(self.save_path, mode="a", index=False, header=write_head)

    def set_algorithm(self, algorithm, compute_fft=False):
        self.algorithm = algorithm
        self.compute_fft = compute_fft

    def set_data(self, data, label, sample_weight=None):
        self.data = data
        self.label = LabelEncoder().fit_transform(label)
        self.sample_weight = sample_weight
        self.data_info = {
            "in_dim": data.shape[1],
            "out_dim": len(np.unique(label))
        }
        self.labels = np.unique(self.label)

    def set_split_func(self, mode="cv", cv=5, split=0.3, sp_path=None, rep=1):
        self.sp_path = sp_path
        if mode == "cv":
            split_func = RepeatedStratifiedKFold(n_splits=cv, n_repeats=rep)
        elif mode == "split":
            split_func = StratifiedShuffleSplit(n_splits=rep, test_size=split)
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

    def _rep_fit_predict(self, split_func):
        results = []
        for idx, (train_idx, test_idx) in tqdm(enumerate(split_func.split(self.data, self.label))):
            if self.verbose:
                print("[ClsFrameWork IDX: {:5d}]".format(idx))
            x_train, x_test = self.data[train_idx, :], self.data[test_idx, :]
            y_train, y_test = self.label[train_idx], self.label[test_idx]

            ans = dict(
                {"idx": idx, "train_idx": train_idx, "test_idx": test_idx},
                **self._fit_predict(x_train, y_train, x_test, y_test)
            )
            # print([(k, v) for k, v in ans.items() if "acc" in k or "bca" in k or "cc" in k])
            # exit()
            results.append(ans)
            if self.verbose:
                print("CurIDX ACC: {:.4f}, BCA: {:.4f}".format(ans["acc"], ans["bca"]))
        return merge_list_of_dicts(results)

    def _fit_predict(self, x_train, y_train, x_test, y_test):
        ss = StandardScaler()
        x_train = ss.fit_transform(x_train)
        x_test = ss.transform(x_test)

        self.algorithm.fit(x_train, y_train)
        y_pred = self.algorithm.predict(x_test)
        if y_pred.ndim != 1:
            y_pred = y_pred.argmax(axis=-1)
        acc = accuracy_score(y_test, y_pred, sample_weight=self.sample_weight)
        bca = balanced_accuracy_score(y_test, y_pred, sample_weight=self.sample_weight)

        if self.compute_fft:
            fft = get_fft_energy(self.algorithm, 0.05, 300, x_train, 300)
            sum_fft = fft.shape[0] + fft.shape[1]
            fft = np.sum(fft**2, axis=(0, 1))
            refit_ans = {}
            for metname, met in [("bca", balanced_accuracy_score), ("acc", accuracy_score)]:
                refit_results = self.algorithm.test_refit_metrics(x_test, y_test, met)
                for n, m in refit_results:
                    refit_ans["{}_{}".format(n, metname)] = m
        else:
            fft = 1
            sum_fft = 1
            refit_ans = {"ridge_acc":0, "ridge_bca":0, "LR_acc":0, "LR_bca":0}
        return {"y_pred": y_pred, "y_test": y_test, "acc": acc, "bca": bca, "fft": fft, "sum_fft": sum_fft, **refit_ans}

    def print_results(self, metrics=None):
        if metrics is None:
            metrics = ["acc", "bca"]
        print("Total Run: {}".format(self.run_info["idx"]))
        for m in metrics:
            print("{}: {:.4f}±{:.4f}".format(
                m.upper(), np.mean(self.run_info[m]), np.std(self.run_info[m]))
            )