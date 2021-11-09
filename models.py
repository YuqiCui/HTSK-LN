import numbers

import torch
import torch.nn as nn
import torch.nn.functional as functional


class Affine(nn.Module):
    def __init__(self, normalized_shape):
        super(Affine, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.weight = nn.Parameter(torch.ones(*normalized_shape))
        self.bias = nn.Parameter(torch.zeros(*normalized_shape))

    def forward(self, X):
        return X * self.weight + self.bias

    def reset_parameters(self):
        torch.nn.init.constant_(self.weight, 1)
        torch.nn.init.constant_(self.bias, 0)


class HTSK(nn.Module):
    def __init__(self, in_dim, out_dim, n_rules, order=1, **params):
        super(HTSK, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_rules = n_rules
        self.order = order
        self.params = params

        self.build_model()

    def reset_parameters(self):
        if hasattr(self, "init_center_"):
            self.centers.data[...] = torch.FloatTensor(self.init_center_)
        else:
            nn.init.normal_(self.centers, 0, 1)

        nn.init.constant_(self.sigmas, self.sigmas_init_scale)
        for module in self.reset_modules:
            module.reset_parameters()

    def init_centers(self, centers):
        self.centers.data[...] = torch.FloatTensor(centers)
        self.init_center_ = centers

    def build_model(self):
        self.sigmas_init_scale = self.params.get("sigma_scale", 1.0)
        self.centers = nn.Parameter(torch.normal(0, 1, size=[self.in_dim, self.n_rules]), requires_grad=True)
        self.sigmas = nn.Parameter(torch.ones([self.in_dim, self.n_rules]) * self.sigmas_init_scale, requires_grad=True)
        self.h = 0.5
        self.eps = 1e-8

        self.reset_modules = []

        # consequent parameters
        if self.order == 0:
            self.cons = nn.Linear(self.n_rules, self.out_dim)
        elif self.order == 1:
            self.cons = nn.Linear((self.in_dim + 1) * self.n_rules, self.out_dim)
        self.reset_modules.append(self.cons)

        if self.params.get("input_bn", False):
            self.input_bn = nn.BatchNorm1d(self.in_dim)
            self.reset_modules.append(self.input_bn)

        if self.params.get("input_bn_only_affine", False):
            self.input_bn_only_affine = Affine(self.in_dim)
            self.reset_modules.append(self.input_bn_only_affine)

        if self.params.get("bn", False):
            self.bn = nn.BatchNorm1d(self.n_rules)
            self.reset_modules.append(self.bn)

        ln_mode = self.params.get("ln", False)
        if callable(ln_mode):
            self.ln = ln_mode
            self.reset_modules.append(self.ln)

        elif ln_mode:
            self.ln = nn.LayerNorm(self.n_rules)
            self.reset_modules.append(self.ln)

        if self.params.get("relu", False):
            self.relu = nn.ReLU()

    def l2_loss(self):
        return torch.mean(self.cons.weight ** 2)

    def forward(self, X, with_frs=False, output_xp=False):
        frs = torch.mean(
            -(X.unsqueeze(dim=2) - self.centers) ** 2 * (self.h / self.sigmas ** 2 + self.eps), dim=1
        )
        frs = functional.softmax(frs, dim=1)  # firing level, [n_batch, n_rules]
        raw_frs = frs

        if hasattr(self, "input_bn"):
            X = self.input_bn(X)

        if hasattr(self, "input_bn_only_affine"):
            X = self.input_bn_only_affine(X)

        if hasattr(self, "ln"):
            frs = self.ln(frs)

        if hasattr(self, "bn"):
            frs = self.bn(frs)

        if hasattr(self, "relu"):
            frs = self.relu(frs)

        if self.order == 1:
            X = X.unsqueeze(dim=1).expand([X.size(0), self.n_rules, X.size(1)])  # [n_batch, n_rules, in_dim]
            X = X * frs.unsqueeze(dim=2)
            X = X.view([X.size(0), -1])
            X = torch.cat([X, frs], dim=1)
        else:
            X = frs

        if output_xp:
            return X

        output = self.cons(X)
        if not with_frs:
            return output
        else:
            return output, raw_frs, frs

    def output_xp(self, X):
        self.eval()
        with torch.no_grad():
            return self(X, output_xp=True)