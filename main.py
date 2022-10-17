import argparse
import os

import numpy as np
import pandas as pd
import pytsk.gradient_descent as pgd  # install pytsk by: pip install scikit-pytsk
import torch.optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from archs.fuzzy import HTSK
from archs.optims import AdaBound
from conf import training_epochs
from frameworks.data import load_reg_data
from frameworks.regression_loss_curve import FrameWork

parser = argparse.ArgumentParser()
# ------- exp setup ----------
parser.add_argument("--data", type=str, default="data", help="Dataset")
parser.add_argument("--optim", type=str, default="adam", help="adam, sgd, adabound")
parser.add_argument("--gpu", type=str, default="cuda:0", help="GPU device, cuda:n or cpu")
parser.add_argument("--rep", type=int, default=10, help="Repeat times")
parser.add_argument("--seed", default=1447, type=int, help="random seed")
parser.add_argument("--overwrite", default=1, type=int, help="Overwrite")
parser.add_argument("--sigma", default=-1, type=float)
parser.add_argument("--val", default=0.1, type=float, help="val size")

# ------- fuzzy system params --------
parser.add_argument("--n_rules", type=int, default=30, help="Num. of rules")
parser.add_argument('--consbn', dest='consbn', action='store_true')
parser.add_argument('--no-consbn', dest='consbn', action='store_false')
parser.set_defaults(consbn=False)
parser.add_argument('--ln', dest='ln', action='store_true')
parser.add_argument('--no-ln', dest='ln', action='store_false')
parser.set_defaults(ln=False)
parser.add_argument('--relu', dest='relu', action='store_true')
parser.add_argument('--no-relu', dest='relu', action='store_false')
parser.set_defaults(relu=False)

# ------- training params --------------
parser.add_argument("--epochs", default=1500, type=int)
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
parser.add_argument("--wd", default=1e-3, type=float, help="Weight decay")
parser.add_argument("--ur", default=0, type=float, help="UR Weight")
parser.add_argument("--ur_tau", default=0.5, type=float, help="UR param tau")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

model_name = "HTSK"
if args.ln:
    model_name += "-LN"
if args.relu:
    model_name += "-ReLU"
if args.consbn:
    model_name += "-ConsBN"
if args.ur > 0:
    model_name += "-UR{}".format(args.ur)

data_format = "data/regression/{}.npz"
data_split_format = "data/splits_regression/{}.npz"
save_root = "res/main/{}".format(args.optim)
pkl_root = "ckpt/main/{}/{}".format(args.optim, args.data)

if not os.path.exists(save_root):
    os.makedirs(save_root)

if not os.path.exists(pkl_root):
    os.makedirs(pkl_root)

save_path = os.path.join(save_root, "{}_{}_r{}_wd{}.csv".format(args.data, model_name, args.n_rules, args.wd))

print("CSV save path: {}".format(save_path))
print("PKL save path: {}".format(pkl_root))

center_param_name = ["centers", "sigmas", "sig1", "sig2"]
sigma_scale_list = [0.1, 0.3, 0.5, 0.7, 1.0] if args.sigma <= 0 else [args.sigma]


def get_optim(model):
    center_params, other_params = [], []
    for n, p in model.named_parameters():
        if n in center_param_name:
            center_params.append(p)
        else:
            other_params.append(p)
    if args.optim == "adam":
        optimizer = torch.optim.Adam(
            [
                {'params': center_params},
                {'params': other_params, 'weight_decay': args.wd},
            ], lr=args.lr, weight_decay=0
        )
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(
            [
                {'params': center_params},
                {'params': other_params, 'weight_decay': args.wd},
            ], lr=args.lr, weight_decay=0, momentum=0.9
        )
    elif args.optim == "adabound":
        optimizer = AdaBound(
            [
                {'params': center_params},
                {'params': other_params, 'weight_decay': args.wd},
            ], lr=args.lr, weight_decay=0
        )
    else:
        raise ValueError("Wrong type of args.optim")
    return optimizer


def train():
    if os.path.exists(save_path):
        print("Save path exists: {}".format(save_path))
        if args.overwrite == 0:
            exit()

    data, label = load_reg_data(data_format.format(args.data))

    label = label.reshape([-1, 1])
    label = StandardScaler().fit_transform(label)
    n_samples = data.shape[0]

    if args.data in training_epochs:
        epochs = training_epochs[args.data]
    else:
        epochs = args.epochs

    if n_samples < 5000:
        batch_size = 64
    elif n_samples < 1000:
        batch_size = 64
    else:
        batch_size = 512

    print("Training epochs: {}, Training batch size".format(epochs, batch_size))

    split_path = data_split_format.format(args.data)
    if not os.path.exists(split_path):
        indices = np.arange(n_samples)
        x_train, x_test, y_train, y_test, train_idx, test_idx = train_test_split(
            data, label, indices, test_size=0.3, random_state=1447)
        np.savez(split_path, train_idx=train_idx, test_idx=test_idx)
    else:
        f = np.load(split_path)
        train_idx, test_idx = f['train_idx'], f['test_idx']
        x_train, x_test, y_train, y_test = data[train_idx], data[test_idx], label[train_idx], label[test_idx]

    print("run on {}, samples: {}, features: {}, label Dist: {:.4f}/{:.4f}".format(
        args.data, data.shape[0], data.shape[1], np.mean(label), np.std(label))
    )

    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

    init_center = pgd.antecedent_init_center(x_train, n_rule=args.n_rules, engine="faiss", n_init=30)
    dfs = []

    for i in range(args.rep):
        for sigma_scale in sigma_scale_list:
            print("[REP {:2d}] Sigma: {:1f}".format(i, sigma_scale))
            addl_info = {
                "batch_size": batch_size, "data_name": args.data,
                "n_rep": i, "train_size": len(train_idx),
                "test_size": len(test_idx)
            }
            frame = FrameWork(
                grad_clip=100, verbose=1, batch_size=batch_size, device=args.gpu, val_size=args.val
            )
            frame.set_data(x_train, y_train, x_test, y_test)

            model = HTSK(
                in_dim=x_train.shape[1], n_rules=args.n_rules,
                out_dim=1, ln=args.ln, relu=args.relu, input_bn=args.consbn,
                sigma_scale=sigma_scale
            )
            model.init_centers(init_center)

            infos = frame.train(
                    model,
                    algo_name="{}-sig{}".format(model_name, sigma_scale),
                    epochs=epochs,
                    optimizer=get_optim(model),
                    addl_info=addl_info,
                    save_path=pkl_root,
                    ur_weight=args.ur,
                    ur_c=args.ur_tau,
            )

            df = pd.DataFrame(infos)
            df["sig"] = sigma_scale
            df["batch_size"] = batch_size
            df["algo_name"] = model_name

            dfs.append(df)
            pd.concat(dfs).to_csv(save_path)


if __name__ == '__main__':
    train()
