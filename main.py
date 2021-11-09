import itertools
import os
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from training import FrameWork, get_optim
from archs.fuzzy import HTSK, TSK, PreHTSK
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans


def load_reg_data(path):
    f = np.load(path)
    return f['data'], f['label']


parse_args = sys.argv[1:]
RUN_DATASET = list(map(int, parse_args[1:]))

DATA_FORMAT = "{}.npz"
SPLIT_FORMAT = "{}.npz"
N_RULES = 30
N_REP = 10
DEVICE = "cuda:0"
SAVEPATH_FORMAT = "{}.csv"
print(SAVEPATH_FORMAT)
MODEL_PKL_FORMAT = "ckpt/{}"

center_param_name1 = ["centers", "sigmas", "sig1", "sig2"]


def train():
    if not os.path.exists("ckpt"):
        os.mkdir("ckpt")
    for data_idx, data_name in enumerate([
        "Puma32h",
    ]):
        print("="*30)
        print("+"*30)

        if not os.path.exists("ckpt/sigma_relu_adam/{}".format(data_name)):
            os.mkdir("ckpt/sigma_relu_adam/{}".format(data_name))

        EPOCHS = 3000 if data_name in ["YearPredictionMSD_small", "Pole", "abalone"] else 1500

        data, label = load_reg_data(DATA_FORMAT.format(data_name))
        label = label.reshape([-1, 1])
        label = StandardScaler().fit_transform(label)
        n_samples = data.shape[0]

        split_path = SPLIT_FORMAT.format(data_name)
        if not os.path.exists(split_path):
            indices = np.arange(n_samples)
            x_train, x_test, y_train, y_test, train_idx, test_idx = train_test_split(
                data, label, indices, test_size=0.3, random_state=1447)
            np.savez(split_path, train_idx=train_idx, test_idx=test_idx)
        else:
            f = np.load(split_path)
            train_idx, test_idx = f['train_idx'], f['test_idx']
            x_train, x_test, y_train, y_test = data[train_idx], data[test_idx], label[train_idx], label[test_idx]

        print("run on {}, samples: {}, features: {}, label Dist: {:.4f}/{:.4f}".format(data_name, data.shape[0], data.shape[1], np.mean(label), np.std(label)))

        ss = StandardScaler()
        x_train = ss.fit_transform(x_train)
        x_test = ss.transform(x_test)

        km = KMeans(n_clusters=N_RULES)
        km.fit(x_train)
        km_centers = km.cluster_centers_.T

        infos = []

        for i in tqdm(range(N_REP)):
            for batch_size, sigma_scale in itertools.product([512], [0.1, 0.3, 0.5, 0.7, 1.0]):
                if n_samples < 5000:
                    batch_size = 64
                    EPOCHS = 3000

                addl_info = {
                    "batch_size": batch_size, "data_name": data_name,
                    "n_rep": i, "train_size": len(train_idx),
                    "test_size": len(test_idx)
                }
                frame = FrameWork(grad_clip=100, verbose=1, batch_size=batch_size, device=DEVICE)
                frame.set_data(x_train, y_train, x_test, y_test)
                model = HTSK(in_dim=x_train.shape[1], n_rules=N_RULES, out_dim=1, input_bn=False, sigma_scale=sigma_scale)
                model.init_centers(km_centers)
                infos.extend(
                    frame.train(model, algo_name="HTSK-sig{}".format(sigma_scale), epochs=EPOCHS, optimizer=get_optim(model, center_param_name=center_param_name1),
                                addl_info=addl_info,
                                save_path=MODEL_PKL_FORMAT.format(data_name)
                                )

                )

                frame = FrameWork(grad_clip=100, verbose=1, batch_size=batch_size, device=DEVICE)
                frame.set_data(x_train, y_train, x_test, y_test)
                model = HTSK(in_dim=x_train.shape[1], n_rules=N_RULES, out_dim=1, input_bn=True, sigma_scale=sigma_scale)
                model.init_centers(km_centers)
                infos.extend(
                    frame.train(model, algo_name="HTSK-BN-sig{}".format(sigma_scale), epochs=EPOCHS, optimizer=get_optim(model, center_param_name=center_param_name1),
                                addl_info=addl_info, save_path=MODEL_PKL_FORMAT.format(data_name))
                )

                frame = FrameWork(grad_clip=100, verbose=1, batch_size=batch_size, device=DEVICE)
                frame.set_data(x_train, y_train, x_test, y_test)
                model = HTSK(in_dim=x_train.shape[1], n_rules=N_RULES, out_dim=1, ln=True, sigma_scale=sigma_scale)
                model.init_centers(km_centers)
                infos.extend(
                    frame.train(model, algo_name="HTSK-LN-sig{}".format(sigma_scale), epochs=EPOCHS, optimizer=get_optim(model, center_param_name=center_param_name1),
                                addl_info=addl_info, save_path=MODEL_PKL_FORMAT.format(data_name))
                )

                frame = FrameWork(grad_clip=100, verbose=1, batch_size=batch_size, device=DEVICE)
                frame.set_data(x_train, y_train, x_test, y_test)
                model = HTSK(in_dim=x_train.shape[1], n_rules=N_RULES, out_dim=1, ln=True, sigma_scale=sigma_scale, relu=True)
                model.init_centers(km_centers)
                infos.extend(
                    frame.train(model, algo_name="HTSK-LN-relu-sig{}".format(sigma_scale), epochs=EPOCHS, optimizer=get_optim(model, center_param_name=center_param_name1),
                                addl_info=addl_info, save_path=MODEL_PKL_FORMAT.format(data_name))
                )

                df = pd.DataFrame(infos)
                df["sig"] = df.algo_name.map(lambda x: x.split("-sig")[1] if "sig" in x else "1.0")
                df["algo_name"] = df.algo_name.map(lambda x: x.split("-sig")[0] if "sig" in x else x)
                df.to_csv(SAVEPATH_FORMAT.format(data_name))


if __name__ == '__main__':
    train()
