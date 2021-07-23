import torch
import torch.nn as nn
import torch.nn.functional as functional


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

        self.build_model()

    def reset_parameters(self):
        nn.init.normal_(self.centers, 0, 1)
        nn.init.constant_(self.sigmas, 1)
        self.cons.reset_parameters()

        if hasattr(self, "input_bn"):
            self.input_bn.reset_parameters()

        if hasattr(self, "bn"):
            self.bn.reset_parameters()

        if hasattr(self, "ln"):
            self.ln.reset_parameters()

    def build_model(self):
        # antecedent parameters
        self.centers = nn.Parameter(torch.normal(0, 1, size=[self.in_dim, self.n_rules]), requires_grad=True)
        self.sigmas = nn.Parameter(torch.ones([self.in_dim, self.n_rules]), requires_grad=True)
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

    def l2_loss(self):
        return torch.mean(self.cons.weight ** 2)

    def forward_firing_level(self, X, with_norm=False):
        frs = torch.mean(
            -(X.unsqueeze(dim=2) - self.centers) ** 2 * (self.h / self.sigmas ** 2 + self.eps), dim=1
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

    def forward(self, X, with_frs=False):
        frs = torch.mean(
            -(X.unsqueeze(dim=2) - self.centers) ** 2 * (self.h / self.sigmas ** 2 + self.eps), dim=1
        )
        frs = functional.softmax(frs, dim=1)  # firing level, [n_batch, n_rules]

        if hasattr(self, "input_bn"):
            X = self.input_bn(X)

        if hasattr(self, "bn"):
            frs = self.bn(frs)

        if hasattr(self, "ln"):
            frs = self.ln(frs)

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
            return output, frs


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

        self.build_model()

    def reset_parameters(self):
        nn.init.normal_(self.centers, 0, 1)
        nn.init.constant_(self.sigmas, 1)
        self.cons.reset_parameters()

        if hasattr(self, "input_bn"):
            self.input_bn.reset_parameters()

        if hasattr(self, "bn"):
            self.bn.reset_parameters()

        if hasattr(self, "ln"):
            self.ln.reset_parameters()


    def build_model(self):
        # antecedent parameters
        self.centers = nn.Parameter(torch.normal(0, 1, size=[self.in_dim, self.n_rules]), requires_grad=True)
        self.sigmas = nn.Parameter(torch.ones([self.in_dim, self.n_rules]), requires_grad=True)
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
            -(X.unsqueeze(dim=2) - self.centers) ** 2 * (self.h / self.sigmas ** 2 + self.eps), dim=1
        )
        frs = functional.softmax(frs, dim=1)  # firing level, [n_batch, n_rules]
        return frs

    def forward(self, X):
        frs = torch.sum(
            -(X.unsqueeze(dim=2) - self.centers) ** 2 * (self.h / self.sigmas ** 2 + self.eps), dim=1
        )
        frs = functional.softmax(frs, dim=1)  # firing level, [n_batch, n_rules]

        if hasattr(self, "input_bn"):
            X = self.input_bn(X)

        if hasattr(self, "bn"):
            frs = self.bn(frs)

        if hasattr(self, "ln"):
            frs = self.ln(frs)

        if hasattr(self, "dropout"):
            frs = self.dropout(frs)
            frs /= torch.sum(frs, dim=1, keepdim=True)

        if self.order == 1:
            X = X.unsqueeze(dim=1).expand([X.size(0), self.n_rules, X.size(1)])  # [n_batch, n_rules, in_dim]
            X = X * frs.unsqueeze(dim=2)
            X = X.view([X.size(0), -1])
            X = torch.cat([X, frs], dim=1)
        else:
            X = frs

        output = self.cons(X)
        return output, frs