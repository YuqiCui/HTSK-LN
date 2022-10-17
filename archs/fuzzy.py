import numbers

import torch
import torch.nn as nn
import torch.nn.functional as functional

from archs.utils import KeepTopK, DropRule
from .layer_norm import LayerNormImpl


class AdaNorm(nn.Module):
    def __init__(self, size=1, C=0.8, k=0.1):
        super(AdaNorm, self).__init__()
        self.ln = nn.LayerNorm(size, elementwise_affine=False)
        self.C = C
        self.k = k

    def forward(self, X):
        X = self.ln(X)
        X = self.C * (1 - self.k * X) * X
        return X

    def reset_parameters(self):
        self.ln.reset_parameters()


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
        assert self.order in [0, 1], "Order must be 0 or 1."
        self.params = params
        self.params_bak = self.params.copy()
        self.sigmas_init_scale = self.params.pop("sigma_scale", 1)
        self.build_model()

        self.ln_stat = 1

    def reset_parameters(self):
        if hasattr(self, "init_center_"):
            self.centers.data[...] = torch.FloatTensor(self.init_center_)
        else:
            nn.init.normal_(self.centers, 0, 1)
        nn.init.constant_(self.sigmas, self.sigmas_init_scale)
        self.cons.reset_parameters()

        if hasattr(self, "input_bn"):
            self.input_bn.reset_parameters()

        if hasattr(self, "bn"):
            self.bn.reset_parameters()

        if hasattr(self, "ln"):
            self.ln.reset_parameters()

    def init_centers(self, centers):
        self.centers.data[...] = torch.FloatTensor(centers)
        self.init_center_ = centers

    def build_model(self):
        # antecedent parameters
        self.centers = nn.Parameter(torch.normal(0, 1, size=[self.in_dim, self.n_rules]), requires_grad=True)
        self.sigmas = nn.Parameter(torch.ones([self.in_dim, self.n_rules]) * self.sigmas_init_scale, requires_grad=True)
        self.h = 0.5
        self.eps = 1e-8

        # consequent parameters
        if self.order == 0:
            self.cons = nn.Linear(self.n_rules, self.out_dim)
        elif self.order == 1:
            self.cons = nn.Linear((self.in_dim + 1) * self.n_rules, self.out_dim)

        if self.params.pop("droprule", False):
            self.droprule = DropRule(p=0.5)

        # other params
        if self.params.pop("bn", False):
            self.bn = nn.BatchNorm1d(self.n_rules)

        if self.params.pop("fake_cbn", False):
            self.fake_cbn = Affine(self.in_dim)

        if self.params.pop("input_bn", False):
            self.input_bn = nn.BatchNorm1d(self.in_dim)

        if self.params.pop("topk", False):
            self.topk = KeepTopK(k=self.n_rules // 2)

        ln = self.params.pop("ln", False)
        if ln:
            adanorm_c = self.params.pop("adanorm_c", 0)
            if adanorm_c > 0:
                self.ln = LayerNormImpl(
                    self.n_rules,
                    mode="adanorm",
                    adanorm_scale=adanorm_c
                )

            elif self.params.pop("simple", False):
                self.ln = nn.LayerNorm(
                    self.n_rules, elementwise_affine=False
                )
                # print(self.ln)
            elif callable(ln):
                self.ln = ln
            else:
                self.ln = nn.LayerNorm(
                    self.n_rules, elementwise_affine=True
                )
            # print(self.ln)

        if self.params.pop("relu", False):
            self.relu = nn.ReLU()

        if self.params.pop("fake_ln", False):
            self.fake_ln = Affine(self.n_rules)

    def l2_loss(self):
        return torch.mean(self.cons.weight ** 2)

    def forward_firing_level(self, X, with_norm=False):
        frs = torch.mean(
            -(X.unsqueeze(dim=2) - self.centers) ** 2 * (self.h / (self.sigmas ** 2 + self.eps)), dim=1
        )
        frs = functional.softmax(frs, dim=1)  # firing level, [n_batch, n_rules]
        if not with_norm:
            return frs
        else:
            if hasattr(self, "bn"):
                frs = self.bn(frs)

            if hasattr(self, "ln"):
                frs = self.ln(frs)
            return frs

    def output_xp(self, X):
        with torch.no_grad():
            frs = torch.mean(
                -(X.unsqueeze(dim=2) - self.centers) ** 2 * (self.h / self.sigmas ** 2 + self.eps), dim=1
            )

            if hasattr(self, "droprule"):
                frs = self.droprule(frs)
                print("!")

            frs = functional.softmax(frs, dim=1)  # firing level, [n_batch, n_rules]
            if hasattr(self, "input_bn"):
                X = self.input_bn(X)

            if hasattr(self, "input_bn"):
                X = self.input_bn(X)

            if hasattr(self, "fake_cbn"):
                X = self.fake_cbn(X)

            if hasattr(self, "bn"):
                frs = self.bn(frs)

            if hasattr(self, "ln"):
                frs = self.ln(frs)

            if hasattr(self, "fake_ln"):
                frs = self.fake_ln(frs)

            if hasattr(self, "relu"):
                frs = self.relu(frs)

            if self.order == 1:
                X = X.unsqueeze(dim=1).expand([X.size(0), self.n_rules, X.size(1)])  # [n_batch, n_rules, in_dim]
                X = X * frs.unsqueeze(dim=2)
                X = X.view([X.size(0), -1])
                X = torch.cat([X, frs], dim=1)
            else:
                X = frs

            return X

    def forward(self, X, with_frs=False):
        frs = torch.mean(
            -(X.unsqueeze(dim=2) - self.centers) ** 2 * (self.h / self.sigmas ** 2 + self.eps), dim=1
        )
        frs = functional.softmax(frs, dim=1)  # firing level, [n_batch, n_rules]
        raw_frs = frs
        # with torch.no_grad():
        #     print(torch.mean(frs, dim=1).squeeze(), torch.std(frs, dim=1).squeeze())
        #     print("=" * 50)

        if hasattr(self, "input_bn"):
            X = self.input_bn(X)

        if hasattr(self, "fake_cbn"):
            X = self.fake_cbn(X)

        if hasattr(self, "bn"):
            frs = self.bn(frs)

        if hasattr(self, "ln"):
            frs = self.ln(frs)

        if hasattr(self, "fake_ln"):
            frs = self.fake_ln(frs)

        if hasattr(self, "relu"):
            frs = self.relu(frs)

        with torch.no_grad():
            self.ln_stat = 0.9 * self.ln_stat + 0.1 * torch.std(frs, dim=1).mean().item()

        if self.order == 1:
            X = X.unsqueeze(dim=1).expand([X.size(0), self.n_rules, X.size(1)])  # [n_batch, n_rules, in_dim]
            X = X * frs.unsqueeze(dim=2)
            X = X.view([X.size(0), -1])
            X = torch.cat([X, frs], dim=1)
        else:
            X = frs

        output = self.cons(X)
        if not with_frs:
            return output
        else:
            return output, raw_frs, frs


class TSK(nn.Module):
    def __init__(self, in_dim, out_dim, n_rules, order=1, **params):
        super(TSK, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_rules = n_rules
        self.order = order
        assert self.order in [0, 1], "Order must be 0 or 1."
        self.params = params
        self.params_bak = self.params.copy()
        self.sigmas_init_scale = self.params.pop("sigma_scale", 1)
        self.ln_stat = 0
        self.build_model()

    def reset_parameters(self):
        if hasattr(self, "init_center_"):
            self.centers.data[...] = torch.FloatTensor(self.init_center_)
        else:
            nn.init.normal_(self.centers, 0, 1)

        nn.init.constant_(self.sigmas, self.sigmas_init_scale)
        self.cons.reset_parameters()

        if hasattr(self, "input_bn"):
            self.input_bn.reset_parameters()

        if hasattr(self, "bn"):
            self.bn.reset_parameters()

        if hasattr(self, "ln"):
            self.ln.reset_parameters()

    def init_centers(self, centers):
        self.centers.data[...] = torch.FloatTensor(centers)
        self.init_center_ = centers

    def build_model(self):
        # antecedent parameters
        self.centers = nn.Parameter(torch.normal(0, 1, size=[self.in_dim, self.n_rules]), requires_grad=True)
        self.sigmas = nn.Parameter(torch.ones([self.in_dim, self.n_rules]) * self.sigmas_init_scale, requires_grad=True)
        self.h = 0.5
        self.eps = 1e-8

        # consequent parameters
        if self.order == 0:
            self.cons = nn.Linear(self.n_rules, self.out_dim)
        elif self.order == 1:
            self.cons = nn.Linear((self.in_dim + 1) * self.n_rules, self.out_dim)

        # other params
        if self.params.pop("bn", False):
            self.bn = nn.BatchNorm1d(self.n_rules)

        if self.params.pop("input_bn", False):
            self.input_bn = nn.BatchNorm1d(self.in_dim)

        if self.params.pop("ln", False):
            self.ln = nn.LayerNorm(self.n_rules)

        dropp = self.params.pop("droprate", 0)
        if dropp > 0:
            self.dropout = nn.Dropout(p=dropp)

    def l2_loss(self):
        return torch.mean(self.cons.weight ** 2)

    def forward_firing_level(self, X):
        frs = torch.sum(
            -(X.unsqueeze(dim=2) - self.centers) ** 2 * (self.h / (self.sigmas ** 2 + self.eps)), dim=1
        )
        frs = functional.softmax(frs, dim=1)  # firing level, [n_batch, n_rules]
        return frs

    def forward(self, X, with_frs=False):
        frs = torch.sum(
            -(X.unsqueeze(dim=2) - self.centers) ** 2 * (self.h / (self.sigmas ** 2 + self.eps)), dim=1
        )
        frs = functional.softmax(frs, dim=1)  # firing level, [n_batch, n_rules]
        raw_frs = frs
        if hasattr(self, "input_bn"):
            X = self.input_bn(X)

        if hasattr(self, "bn"):
            frs = self.bn(frs)

        if hasattr(self, "ln"):
            frs = self.ln(frs)

        if hasattr(self, "dropout"):
            frs = self.dropout(frs)
            frs /= torch.sum(frs, dim=1, keepdim=True)

        with torch.no_grad():
            self.ln_stat = 0.9 * self.ln_stat + 0.1 * torch.std(frs, dim=1).mean().item()

        if self.order == 1:
            X = X.unsqueeze(dim=1).expand([X.size(0), self.n_rules, X.size(1)])  # [n_batch, n_rules, in_dim]
            X = X * frs.unsqueeze(dim=2)
            X = X.view([X.size(0), -1])
            X = torch.cat([X, frs], dim=1)
        else:
            X = frs

        output = self.cons(X)
        if not with_frs:
            return output
        else:
            return output, raw_frs, frs

    def output_xp(self, X):
        with torch.no_grad():
            frs = torch.mean(
                -(X.unsqueeze(dim=2) - self.centers) ** 2 * (self.h / self.sigmas ** 2 + self.eps), dim=1
            )
            frs = functional.softmax(frs, dim=1)  # firing level, [n_batch, n_rules]
            if hasattr(self, "input_bn"):
                X = self.input_bn(X)

            if hasattr(self, "input_bn"):
                X = self.input_bn(X)

            if hasattr(self, "fake_cbn"):
                X = self.fake_cbn(X)

            if hasattr(self, "bn"):
                frs = self.bn(frs)

            if hasattr(self, "ln"):
                frs = self.ln(frs)

            if hasattr(self, "fake_ln"):
                frs = self.fake_ln(frs)

            if hasattr(self, "relu"):
                frs = self.relu(frs)

            if self.order == 1:
                X = X.unsqueeze(dim=1).expand([X.size(0), self.n_rules, X.size(1)])  # [n_batch, n_rules, in_dim]
                X = X * frs.unsqueeze(dim=2)
                X = X.view([X.size(0), -1])
                X = torch.cat([X, frs], dim=1)
            else:
                X = frs

            return X


class PreHTSK(nn.Module):
    def __init__(self, in_dim, out_dim, n_rules, order=1, **params):
        super(PreHTSK, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_rules = n_rules
        self.order = order
        assert self.order in [0, 1], "Order must be 0 or 1."
        self.params = params
        self.params_bak = self.params.copy()
        self.sigmas_init_scale = self.params.pop("sigma_scale", 1)
        self.build_model()

        self.ln_stat = 1

    def reset_parameters(self):
        if hasattr(self, "init_center_"):
            self.centers.data[...] = torch.FloatTensor(self.init_center_)
        else:
            nn.init.normal_(self.centers, 0, 1)
        nn.init.constant_(self.sigmas, self.sigmas_init_scale)
        self.cons.reset_parameters()

        self.ln.reset_parameters()
        self.fln.reset_parameters()
        self.input_bn.reset_parameters()

    def init_centers(self, centers):
        self.centers.data[...] = torch.FloatTensor(centers)
        self.init_center_ = centers

    def build_model(self):
        # antecedent parameters
        self.centers = nn.Parameter(torch.normal(0, 1, size=[self.in_dim, self.n_rules]), requires_grad=True)
        self.sigmas = nn.Parameter(torch.ones([self.in_dim, self.n_rules]) * self.sigmas_init_scale, requires_grad=True)
        self.h = 0.5
        self.eps = 1e-8

        # consequent parameters
        if self.order == 0:
            self.cons = nn.Linear(self.n_rules, self.out_dim)
        elif self.order == 1:
            self.cons = nn.Linear((self.in_dim + 1) * self.n_rules, self.out_dim)

        self.ln = nn.LayerNorm(
            (self.in_dim + 1) * self.n_rules, elementwise_affine=True
        )
        self.input_bn = nn.BatchNorm1d(self.in_dim)

        self.fln = nn.LayerNorm(self.n_rules, elementwise_affine=True)
        if self.params.pop("relu", False):
            self.relu = nn.ReLU()

    def l2_loss(self):
        return torch.mean(self.cons.weight ** 2)

    def forward_firing_level(self, X, with_norm=False):
        frs = torch.mean(
            -(self.ln(X).unsqueeze(dim=2) - self.centers) ** 2 * (self.h / (self.sigmas ** 2 + self.eps)), dim=1
        )
        frs = functional.softmax(frs, dim=1)  # firing level, [n_batch, n_rules]
        return frs

    def output_xp(self, X):
        with torch.no_grad():
            frs = torch.mean(
                -(X.unsqueeze(dim=2) - self.centers) ** 2 * (self.h / self.sigmas ** 2 + self.eps), dim=1
            )
            frs = functional.softmax(frs, dim=1)  # firing level, [n_batch, n_rules]

            if self.order == 1:
                X = X.unsqueeze(dim=1).expand([X.size(0), self.n_rules, X.size(1)])  # [n_batch, n_rules, in_dim]
                X = X * frs.unsqueeze(dim=2)
                X = X.view([X.size(0), -1])
                X = torch.cat([X, frs], dim=1)
            else:
                X = frs

            return X

    def forward(self, X, with_frs=False):
        frs = torch.mean(
            -(X.unsqueeze(dim=2) - self.centers) ** 2 * (self.h / self.sigmas ** 2 + self.eps), dim=1
        )
        frs = functional.softmax(frs, dim=1)  # firing level, [n_batch, n_rules]
        raw_frs = frs

        with torch.no_grad():
            self.ln_stat = 0.9 * self.ln_stat + 0.1 * torch.std(frs, dim=1).mean().item()

        if self.order == 1:
            X = X.unsqueeze(dim=1).expand([X.size(0), self.n_rules, X.size(1)])  # [n_batch, n_rules, in_dim]
            X = X * frs.unsqueeze(dim=2)
            X = X.view([X.size(0), -1])
            X = torch.cat([X, frs], dim=1)
        else:
            X = frs
        X = self.ln(X)
        output = self.cons(X)
        if not with_frs:
            return output
        else:
            return output, raw_frs, frs

if __name__ == '__main__':
    shtsk = Affine(10)
    x = torch.zeros([32, 10])
    shtsk(x)