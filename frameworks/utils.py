import importlib.util

import joblib
import numpy as np
import torch
from collections import defaultdict

from sklearn.cluster import KMeans


def antecedent_init_center(X, y=None, n_rule=2, method="kmean", engine="sklearn", n_init=20):
    """

    This function run KMeans clustering to obtain the :code:`init_center` for :func:`AntecedentGMF() <AntecedentGMF>`.

    Examples
    --------
    >>> init_center = antecedent_init_center(X, n_rule=10, method="kmean", n_init=20)
    >>> antecedent = AntecedentGMF(X.shape[1], n_rule=10, init_center=init_center)


    :param numpy.array X: Feature matrix with the size of :math:`[N,D]`, where :math:`N` is the
        number of samples, :math:`D` is the number of features.
    :param numpy.array y: None, not used.
    :param int n_rule: Number of rules :math:`R`. This function will run a KMeans clustering to
        obtain :math:`R` cluster centers as the initial antecedent center for TSK modeling.
    :param str method: Current version only support "kmean".
    :param str engine: "sklearn" or "faiss". If "sklearn", then the :code:`sklearn.cluster.KMeans()`
        function will be used, otherwise the :code:`faiss.Kmeans()` will be used. Faiss provide a
        faster KMeans clustering algorithm, "faiss" is recommended for large datasets.
    :param int n_init: Number of initialization of the KMeans algorithm. Same as the parameter
        :code:`n_init` in :code:`sklearn.cluster.KMeans()` and the parameter :code:`nredo` in
        :code:`faiss.Kmeans()`.
    """
    def faiss_cluster_center(X, y=None, n_rule=2, n_init=20):
        import faiss
        km = faiss.Kmeans(d=X.shape[1], k=n_rule, nredo=n_init)
        km.train(np.ascontiguousarray(X.astype("float32")))
        centers = km.centroids.T
        return centers

    if method == "kmean":
        if engine == "faiss":
            package_name = "faiss"
            spec = importlib.util.find_spec(package_name)
            if spec is not None:
                center = faiss_cluster_center(X=X, y=y, n_rule=n_rule)
                return center
            else:
                print("Package " + package_name + " is not installed, running scikit-learn KMeans...")
                exit()
        km = KMeans(n_clusters=n_rule, n_init=n_init)
        km.fit(X)
        return km.cluster_centers_.T


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