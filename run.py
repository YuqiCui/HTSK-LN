import numpy as np
from frameworks.data import to_float_tensors, load_reg_data
from sklearn.model_selection import cross_val_score, GridSearchCV, RepeatedKFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, make_scorer, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from archs.fuzzy import HTSK
from frameworks.model_wrapper import URWrapper, ANFISWrapper, Wrapper
from sklearn.datasets import load_iris

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


data, label = load_reg_data("reg_data/yacht.npz")

label = label.reshape([-1, 1])


rmses = []
rmses_refit = []
cnt = 0
for train_idx, test_idx in RepeatedKFold(n_splits=5, n_repeats=4).split(data, label):
    cnt += 1
    x_train, y_train = data[train_idx, :], label[train_idx]
    x_test, y_test = data[test_idx, :], label[test_idx]

    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
    }
    search = GridSearchCV(SVR(), param_grid, n_jobs=10, cv=5, verbose=0)
    search.fit(x_train,  y_train.ravel())
    y_pred = search.predict(x_test)
    print("SVM RMSE: {:.4f}".format(np.sqrt(mean_squared_error(y_test, y_pred))))

    search = Wrapper(
            HTSK(data.shape[1], 1, 30, ln=True),  # HTSK-LN model
            # HTSK(data.shape[1], 1, 30, bn=True),  # HTSK-BN model
            # HTSK(data.shape[1], 1, 30, input_bn=True),  # HTSK-CBN model
            # HTSK(data.shape[1], 1, 30),  # HTSK model

            optimizer_type="adabound",
            loss_type="mse",
            val_metric="mse",
            init_lr=0.01,
            weight_decay=1e-5,
            verbose=0,
            grad_clip=100,
            save_path="test3.pkl",
        )
    search.fit(x_train, y_train)
    y_pred = search.predict(x_test)
    rmses.append(rmse(y_test, y_pred))
    y_pred_refit = search.predict_refit(x_test)
    rmses_refit.append(rmse(y_test, y_pred_refit))
    print("rep: {}, RMSE: {:.4f}, Refit RMSE: {:.4f}".format(cnt, rmses[-1], rmses_refit[-1]))
print(rmses, np.mean(rmses))
print(rmses_refit, np.mean(rmses_refit))

