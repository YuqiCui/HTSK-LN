from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.io as sp
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.datasets import load_digits, load_svmlight_file
from tqdm import tqdm

save_to_dir = False


def libsvm_data(name):
    X, y = load_svmlight_file("raw/{}.libsvm".format(name))
    X = np.array(X.todense())
    y = np.array(y)
    return X, y


def standard_r_data(name):
    df = pd.read_csv("raw/{}_R.dat".format(name), sep="\t")
    df.replace("?", np.nan, inplace=True)
    df.dropna(axis=0, inplace=True)
    value = df.values.astype("float32")

    y = value[:, -1]
    X = value[:, :-1]
    return X, y


def process_utk_face():
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torch.utils.data as Data
    cnn = models.resnext50_32x4d(pretrained=True)
    cnn.eval()

    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, x):
            return x

    cnn.fc = Identity()
    cnn.to("cuda:0")

    mean = np.array([0.485, 0.456, 0.406])[np.newaxis, :, np.newaxis, np.newaxis]
    std = np.array([0.229, 0.224, 0.225])[np.newaxis, :, np.newaxis, np.newaxis]

    mean = torch.from_numpy(mean).float()
    std = torch.from_numpy(std).float()

    image_dir = Path("UTKFace")  # can be downloaded at: https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUQyY1ZDRUE?resourcekey=0-01Pth1hq20K4kuGVkp3oBw
    img_size = 224

    out_genders = []
    out_ages = []
    out_imgs = []

    for i, image_path in enumerate(tqdm(image_dir.glob("*.jpg"))):
        image_name = image_path.name  # [age]_[gender]_[race]_[date&time].jpg
        age, gender = image_name.split("_")[:2]
        out_genders.append(int(gender))
        out_ages.append(min(int(age), 100))
        img = cv2.imread(str(image_path))
        img = cv2.resize(img, (img_size, img_size)).transpose([2, 0, 1])
        img = torch.from_numpy(img).float().unsqueeze(0)
        img = (img - mean) / std

        out_imgs.append(img)

        # if i > 5000:
        #     break

    out_imgs = torch.cat(out_imgs, dim=0)
    out_ages = torch.from_numpy(np.array(out_ages)).float().unsqueeze(1)
    dataset = Data.TensorDataset(out_imgs, out_ages)
    loader = Data.DataLoader(

        dataset=dataset,

        batch_size=12,

        shuffle=False,

        num_workers=0,

    )
    cnn_out = []
    cnn_target = []
    for inputs, targets in tqdm(loader):
        inputs = inputs.to("cuda:0")
        targets = targets.to("cuda:0")
        out = cnn(inputs)
        cnn_out.append(out.detach().cpu().numpy())
        cnn_target.append(targets.detach().cpu().numpy())

    cnn_out = np.concatenate(cnn_out, axis=0)
    cnn_target = np.concatenate(cnn_target, axis=0)
    kpca = PCA(n_components=0.99)
    cnn_out = kpca.fit_transform(cnn_out)

    return cnn_out, cnn_target

# scikit-digits, SD
X, y = load_digits(n_class=10, return_X_y=True, as_frame=False)
if save_to_dir:
    np.savez("processed/scikit_digits.npz", data=X, label=y)

# Space GA, SG
X, y = libsvm_data("space_ga")
if save_to_dir:
    np.savez("processed/space_ga.npz", data=X, label=y)

# Abalone, ABA
X, y = libsvm_data("abalone")
if save_to_dir:
    np.savez("processed/abalone.npz", data=X, label=y)

# Park Motor UPDRS, PM
X, y = standard_r_data("park_motor_UPDRS")
if save_to_dir:
    np.savez("processed/park_motor_UPDRS.npz", data=X, label=y)

# puma32h, PUM
f = sp.loadmat("raw/Puma32h.mat")
data = f['data']
X = data[:, :-1]
y = data[:, -1]
if save_to_dir:
    np.savez("processed/puma.npz", data=X, label=y)

# power plant, PP
data = np.loadtxt("raw/power_plant.txt")
X = data[:, np.arange(4)]
y = data[:, 4]
if save_to_dir:
    np.savez("processed/power_plant.npz", data=X, label=y)

# naval, NAV
df = pd.read_csv("raw/{}.csv".format("naval"))
df.replace("?", np.nan, inplace=True)
df.dropna(axis=0, inplace=True)
value = df.values.astype("float32")

y = value[:, -1]
X = value[:, :-1]
if save_to_dir:
    np.savez("processed/naval.npz", data=X, label=y)

# UTK face, UTK
# Need to download the tar.gz file from https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUQyY1ZDRUE?resourcekey=0-01Pth1hq20K4kuGVkp3oBw
X, y = process_utk_face()
if save_to_dir:
    np.savez("processed/utk_face.npz", data=X, label=y)

# Steel industry, SI
df = pd.read_csv("raw/Steel_industry_data.csv")
ord_enc = OrdinalEncoder()
df['WeekStatus'] = ord_enc.fit_transform(df[["WeekStatus"]])
df['Day_of_week'] = ord_enc.fit_transform(df[["Day_of_week"]])
df['Load_Type'] = ord_enc.fit_transform(df[["Load_Type"]])
X = df.drop('Usage_kWh', axis=1).values
y = df[["Usage_kWh"]]
if save_to_dir:
    np.savez("processed/Steel_industry.npz", data=X, label=y)

# Diamonds, DIA
df = pd.read_csv("raw/diamonds.csv")
cut_class_dict = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5, }
clarity_dict   = {'SI2': 2, 'SI1': 3, 'VS1': 5, 'VS2': 4, 'VVS2': 6, 'VVS1': 7, 'I1': 1, 'IF': 8}
color_dict     = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 5,'D': 5,}
df['cut'] = df['cut'].map(cut_class_dict)
df['clarity'] = df['clarity'].map(clarity_dict)
df['color'] = df['color'].map(color_dict)
X = df.drop('price', axis=1).values  # X use all columns except "Price"
y = df['price'].values  # Y use 'Price' 's values
idx_keep = y < 5000
if save_to_dir:
    np.savez("processed/diamonds.npz", data=X[idx_keep], label=y[idx_keep])

# Microsoft, MIC
X, y = load_svmlight_file("raw/MSLR-WEB10K-fold1/train.txt")  # download it from https://www.microsoft.com/en-us/research/project/mslr/
_, X, _, y = train_test_split(X, y, test_size=50000)
X = X.toarray()
bmax = np.max(X, axis=1)
max_mean = np.mean(bmax)
max_std = np.std(bmax)
keep = abs(bmax - max_mean) < max_std
X = X[keep, :]
y = y[keep]
if save_to_dir:
    np.savez("processed/Microsoft.npz", data=X, label=y)

# Year prediction MSD, YP
df = pd.read_csv("raw/YearPredictionMSD.txt", header=None)  # download it from https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd
cols = df.columns
label = df.iloc[:, 0].values
data = df.iloc[:, 1:].values
_, X, _, y = train_test_split(X, y, test_size=50000)
if save_to_dir:
    np.savez("processed/{}.npz".format("YearPredictionMSD_small"), data=data, label=label)