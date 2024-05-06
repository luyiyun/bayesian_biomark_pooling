import logging

from scipy.special import expit
from numpy import ndarray
from numpy.random import Generator
import torch
from torch import Tensor
from torch.nn.functional import logsigmoid

from ..logger import logger_embp
from .base import EM


EPS = 1e-5
LOGIT_3 = 6.9067548


def ols(
    X: Tensor, Y: Tensor, Z: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    X_des = torch.stack([X, torch.ones_like(X)], dim=1)
    if Z is not None:
        X_des = torch.cat([X_des, Z], dim=1)
    beta, resid, _, _ = torch.linalg.lstsq(X_des, Y)
    return beta, resid


def logistic(
    X: Tensor,
    Y: Tensor,
    Z: Tensor | None = None,
    lr: float = 1.0,
    delta1: float = 1e-3,
    delta2: float = 1e-7,
    max_iter: int = 100,
) -> Tensor:
    X_des = torch.stack([X, torch.ones_like(X)], dim=-1)
    if Z is not None:
        X_des = torch.cat([X_des, Z], dim=-1)
    X_des = X_des.contiguous()

    beta_ = torch.zeros(X_des.shape[1], dtype=torch.double, device=X.device)
    for i in range(max_iter):
        p = (X_des @ beta_).sigmoid()
        grad = X_des.T @ (p - Y)
        hess = torch.einsum("ij,i,ik->jk", X_des, p * (1 - p), X_des)

        beta_delta = lr * torch.linalg.solve(hess, grad)
        beta_ -= beta_delta

        rdiff = (beta_delta.abs() / (beta_.abs() + delta1)).max()
        logger_embp.info(
            f"Init step Newton-Raphson: iter={i+1} diff={rdiff:.4f}"
        )
        if rdiff < delta2:
            break
    else:
        logger_embp.warning(
            f"Init Newton-Raphson (max_iter={max_iter}) doesn't converge"
        )

    return beta_


class TorchEM(EM):
    def __init__(
        self,
        max_iter: int = 100,
        max_iter_inner: int = 100,
        delta1: float = 1e-3,
        delta1_inner: float = 1e-4,
        delta2: float = 1e-4,
        delta2_inner: float = 1e-6,
        delta1_var: float = 1e-2,
        delta2_var: float = 1e-2,
        pbar: bool = True,
        random_seed: int | None | Generator = None,
        device: str = "cuda:0",
    ) -> None:
        super().__init__(
            max_iter=max_iter,
            max_iter_inner=max_iter_inner,
            delta1=delta1,
            delta1_inner=delta1_inner,
            delta1_var=delta1_var,
            delta2=delta2,
            delta2_inner=delta2_inner,
            delta2_var=delta2_var,
            pbar=pbar,
            random_seed=random_seed,
        )
        self._device = torch.device(device)

    @property
    def parameters(self) -> ndarray:
        return self.params_.cpu().numpy()

    @property
    def parameter_history(self) -> ndarray:
        return torch.stack(self.params_hist_).cpu().numpy()

    def prepare(
        self,
        X: ndarray,
        S: ndarray,
        W: ndarray,
        Y: ndarray,
        Z: ndarray | None = None,
    ):
        assert X.shape == S.shape == W.shape == Y.shape
        if Z is not None:
            assert X.shape == Z.shape[:-1]
        assert X.ndim == 1

        self._X = torch.tensor(X, device=self._device, dtype=torch.double)
        self._S = torch.tensor(S, device=self._device, dtype=torch.double)
        self._W = torch.tensor(W, device=self._device, dtype=torch.double)
        self._Y = torch.tensor(Y, device=self._device, dtype=torch.double)
        self._Z = (
            None
            if Z is None
            else torch.tensor(Z, device=self._device, dtype=torch.double)
        )

        # 准备后续步骤中会用到的array，预先计算，节省效率
        self._is_m = self._X.isnan()
        self._is_o = ~self._is_m
        self._ind_m = self._is_m.nonzero().squeeze(-1)
        self._ind_o = self._is_o.nonzero().squeeze(-1)

        self._Xo = self._X[self._ind_o]
        self._Yo = self._Y[self._ind_o]
        self._Wo = self._W[self._ind_o]
        self._Ym = self._Y[self._ind_m]
        self._Wm = self._W[self._ind_m]
        if self._Z is not None:
            self._Zo = self._Z[self._ind_o]
            self._Zm = self._Z[self._ind_m]
        else:
            self._Zo = self._Zm = None

        self._studies, self._ind_inv = torch.unique(
            self._S, return_inverse=True
        )
        self._ind_m_inv = self._ind_inv[self._is_m]

        # the transpose of 1-d array is still 1-d array
        self._ind_S = [
            (self._S == s).nonzero().squeeze(-1) for s in self._studies
        ]
        self._ind_Sm = [
            torch.nonzero((self._S == s) & self._is_m).squeeze(-1)
            for s in self._studies
        ]
        self._ind_So = [
            torch.nonzero((self._S == s) & self._is_o).squeeze(-1)
            for s in self._studies
        ]

        self._n = self._Y.shape[-1]
        self._ns = self._studies.shape[-1]
        self._nz = 0 if self._Z is None else self._Z.shape[-1]
        self._n_o = self._is_o.sum()
        self._n_m = self._is_m.sum()
        self._n_s = torch.tensor(
            [indi.shape[-1] for indi in self._ind_S],
            dtype=torch.double,
            device=self._device,
        )

        self._wbar_s = torch.stack(
            [self._W[ind].mean() for ind in self._ind_S]
        )
        self._wwbar_s = torch.stack(
            [(self._W[ind] ** 2).mean() for ind in self._ind_S]
        )

        self._params_ind = {
            "mu_x": slice(0, 1),
            "sigma2_x": slice(1, 2),
            "a": slice(2, 2 + self._ns),
            "b": slice(2 + self._ns, 2 + 2 * self._ns),
            "sigma2_w": slice(2 + 2 * self._ns, 2 + 3 * self._ns),
        }

        self._Xhat = self._X.clone()
        self._Xhat2 = self._Xhat**2

    def init(self) -> Tensor:
        """初始化参数

        这里仅初始化mu_x,sigma2_x,a,b,sigma2_w，其他和outcome相关的参数需要在子类
        中初始化

        Returns:
            dict: 参数组成的dict
        """
        mu_x = self._Xo.mean(-1, keepdims=True)
        sigma2_x = torch.var(self._Xo, unbiased=True, dim=-1, keepdims=True)

        a, b, sigma2_w = [], [], []
        for ind_so_i in self._ind_So:
            if len(ind_so_i) == 0:
                a.append(0)
                b.append(0)
                sigma2_w.append(1)
                continue

            abi, sigma2_ws_i = ols(self._X[ind_so_i], self._W[ind_so_i])
            a.append(abi[1])
            b.append(abi[0])
            sigma2_w.append(sigma2_ws_i[0])
        a, b, sigma2_w = (
            torch.stack(a),
            torch.stack(b),
            torch.stack(sigma2_w),
        )
        res = torch.cat([mu_x, sigma2_x, a, b, sigma2_w]).contiguous()
        return res

    def calc_rdiff(self, new: Tensor, old: Tensor) -> float:
        return ((old - new).abs() / ((old).abs() + self._delta1)).max().item()


class BinaryEMTorch(TorchEM):

    def __init__(
        self,
        max_iter: int = 100,
        max_iter_inner: int = 100,
        delta1: float = 0.001,
        delta1_inner: float = 0.0001,
        delta2: float = 0.0001,
        delta2_inner: float = 0.000001,
        delta1_var: float = 0.01,
        delta2_var: float = 0.01,
        pbar: bool = True,
        random_seed: int | None | Generator = None,
        device: str = "cuda:0",
        gem: bool = True,
    ) -> None:
        super().__init__(
            max_iter,
            max_iter_inner,
            delta1,
            delta1_inner,
            delta2,
            delta2_inner,
            delta1_var,
            delta2_var,
            pbar,
            random_seed,
            device,
        )
        self._gem = gem

    @property
    def parameter_names(self) -> list:
        return (
            ["mu_x", "sigma_x"]
            + [f"a_{si}" for si in self._studies]
            + [f"b_{si}" for si in self._studies]
            + [f"sigma2_w_{si}" for si in self._studies]
            + ["beta_x"]
            + [f"beta_0_{si}" for si in self._studies]
            + [f"beta_z_{i}" for i in range(self._nz)]
        )

    def prepare(
        self,
        X: ndarray,
        S: ndarray,
        W: ndarray,
        Y: ndarray,
        Z: ndarray | None = None,
    ):
        super().prepare(X, S, W, Y, Z)

        # 用于e-step中的newton algorithm，记录Laplacian Approximation的mu
        self._Xm = torch.zeros(
            self._n_m, device=self._device, dtype=torch.double
        )

        C = torch.zeros(
            (self._n, self._ns), device=self._device, dtype=torch.double
        )
        for i in range(self._ns):
            C[self._ind_S[i], i] = 1
        Xo_des = [self._Xo[:, None], C[self._is_o, :]]  # 360 * 5
        Cm_des = [C[self._is_m, :]]
        if self._Z is not None:
            Xo_des.append(self._Zo)
            Cm_des.append(self._Zm)
        self._Xo_des = torch.cat(Xo_des, 1).contiguous()
        self._Cm_des = torch.cat(Cm_des, 1).contiguous()

        self._params_ind.update(
            {
                "beta_x": slice(2 + 3 * self._ns, 3 + 3 * self._ns),
                "beta_0": slice(3 + 3 * self._ns, 3 + 4 * self._ns),
                "beta_z": slice(3 + 4 * self._ns, 3 + 4 * self._ns + self._nz),
            }
        )
        self._nparams = 3 + 4 * self._ns + self._nz

    def init(self) -> Tensor:
        """初始化权重"""
        params = super().init()
        beta = logistic(self._Xo, self._Yo, self._Zo)
        return torch.cat(
            [params, beta[[0] + [1] * self._ns], beta[2:]]
        ).contiguous()

    def e_step(self, params: Tensor):
        mu_x, sigma2_x, a, b, sigma2_w, beta_x, beta_0 = (
            params[..., self._params_ind[k]]
            for k in [
                "mu_x",
                "sigma2_x",
                "a",
                "b",
                "sigma2_w",
                "beta_x",
                "beta_0",
            ]
        )
        beta_z = (
            params[..., self._params_ind["beta_z"]]
            if self._Z is not None
            else 0.0
        )

        beta_0_m_long = beta_0[self._ind_m_inv]

        # 使用newton-raphson方法得到Laplacian approximation
        # NOTE: 使用Scipy.Newton-CG无法收敛，反而自己的这个每次在3个步骤之内就收敛了
        p_mult = sigma2_w * sigma2_x * beta_x
        p2_mult = p_mult * beta_x
        x_mult = sigma2_x * b**2 + sigma2_w
        const_part = sigma2_w * mu_x - sigma2_x * b * a

        const_m_long = (
            p_mult[self._ind_m_inv] * self._Ym
            + (sigma2_x * b)[self._ind_m_inv] * self._Wm
            + const_part[self._ind_m_inv]
        )
        p_mult_m_long = p_mult[self._ind_m_inv]
        p2_mult_m_long = p2_mult[self._ind_m_inv]
        x_mult_m_long = x_mult[self._ind_m_inv]

        Z_part_m = 0.0 if self._Z is None else self._Zm @ beta_z
        delta_part = beta_0_m_long + Z_part_m

        # 使用上一次迭代的Xm作为初始值
        for i in range(1, self._max_iter_inner + 1):
            p = (self._Xm * beta_x + delta_part).sigmoid()

            xdelta = (
                # self._lr *
                (p_mult_m_long * p + x_mult_m_long * self._Xm - const_m_long)
                / (p2_mult_m_long * p * (1 - p) + x_mult_m_long)
            )
            self._Xm -= xdelta

            rdiff = (
                xdelta.abs() / (self._Xm.abs() + self._delta1_inner)
            ).max()
            logger_embp.info(
                f"E step Newton-Raphson: iter={i} diff={rdiff:.4f}"
            )
            if rdiff < self._delta2_inner:
                break
        else:
            logger_embp.warning(
                f"E step Newton-Raphson (max_iter={self._max_iter_inner})"
                " doesn't converge"
            )

        # 重新计算一次hessian，并计算得到Laplacian Approximation的variance
        p = (self._Xm * beta_x + delta_part).sigmoid()
        self._Vm = (sigma2_w * sigma2_x)[self._ind_m_inv] / (
            p2_mult_m_long * p * (1 - p) + x_mult_m_long
        )
        self._Xm_Sigma = (self._Vm).sqrt()

        self._e_step_update_statistics(
            mu_x=mu_x,
            sigma2_x=sigma2_x,
            beta_x=beta_x,
            a=a,
            b=b,
            sigma2_w=sigma2_w,
            delta_part=delta_part,
        )

    def m_step(self, params: Tensor) -> Tensor:
        vbar = self._Xhat2.mean()
        xbars = torch.stack([self._Xhat[sind].mean() for sind in self._ind_S])
        vbars = torch.stack([self._Xhat2[sind].mean() for sind in self._ind_S])
        wxbars = torch.stack(
            [
                torch.mean(self._W[sind] * self._Xhat[sind])
                for sind in self._ind_S
            ]
        )

        # 更新参数：mu_x,sigma2_x,a,b,sigma2_w
        mu_x = self._Xhat.mean(-1, keepdim=True)
        sigma2_x = vbar - mu_x**2
        b = (wxbars - self._wbar_s * xbars) / (vbars - xbars**2)
        a = self._wbar_s - b * xbars
        sigma2_w = (
            self._wwbar_s
            + vbars * b**2
            + a**2
            - 2 * (a * self._wbar_s + b * wxbars - a * b * xbars)
        )

        # 使用newton-raphson算法更新beta_x,beta_0,beta_z
        # 基于Laplacian Approximation和Importance Sampling的过程会有不同
        beta_all = params[self._params_ind["beta_x"].start :]
        beta_all = self._m_step_update_beta(beta_all)
        return torch.cat(
            [
                mu_x,
                sigma2_x,
                a,
                b,
                sigma2_w,
                beta_all,
            ]
        ).contiguous()


class ISBinaryEMTorch(BinaryEMTorch):

    def __init__(
        self,
        max_iter: int = 100,
        max_iter_inner: int = 100,
        delta1: float = 0.001,
        delta1_inner: float = 0.0001,
        delta2: float = 0.0001,
        delta2_inner: float = 0.000001,
        delta1_var: float = 0.01,
        delta2_var: float = 0.01,
        pbar: bool = True,
        random_seed: int | None | Generator = None,
        device: str = "cuda:0",
        gem: bool = True,
        min_nIS: int = 100,
        max_nIS: int = 5000,
    ) -> None:
        assert max_nIS >= min_nIS
        super().__init__(
            max_iter,
            max_iter_inner,
            delta1,
            delta1_inner,
            delta2,
            delta2_inner,
            delta1_var,
            delta2_var,
            pbar,
            random_seed,
            device,
            gem,
        )
        self._nIS = self._min_nIS = min_nIS
        self._max_nIS = max_nIS

    def _e_step_update_statistics(self, **params):
        beta_x = params["beta_x"]
        mu_x = params["mu_x"]
        sigma2_x = params["sigma2_x"]
        delta_part = params["delta_part"]
        a_m_long = params["a"][self._ind_m_inv]
        b_m_long = params["b"][self._ind_m_inv]
        sigma2_w_m_long = params["sigma2_w"][self._ind_m_inv]

        norm_lap = torch.distributions.Normal(
            loc=self._Xm, scale=self._Xm_Sigma + EPS
        )  # TODO: 这个EPS可能不是必须的

        # 进行IS采样
        self._XIS = norm_lap.sample((self._nIS,))  # N x n_m

        # 计算对应的(normalized) importance weights
        pIS = logsigmoid(
            (2 * self._Ym - 1) * (beta_x * self._XIS + delta_part)
        )
        pIS -= 0.5 * (
            (self._Wm - a_m_long - b_m_long * self._XIS) ** 2 / sigma2_w_m_long
            + (self._XIS - mu_x) ** 2 / sigma2_x
        )
        pIS = pIS - norm_lap.log_prob(self._XIS)
        # NOTE: 尽管归一化因子对于求极值没有贡献，但有助于稳定训练
        self._WIS = torch.softmax(pIS, dim=0)

        if logger_embp.level <= logging.INFO:
            Seff = 1 / (self._WIS**2).sum(dim=0)
            logger_embp.info(
                "Importance effective size "
                + f"is {Seff.mean():.2f}±{Seff.std():.2f}"
            )

        # 计算Xhat和Xhat2, 并讲self._Xm更新为IS计算的后验均值
        self._Xhat[self._is_m] = self._Xm = torch.sum(
            self._XIS * self._WIS, dim=0
        )
        self._Xhat2[self._is_m] = torch.sum(self._XIS**2 * self._WIS, dim=0)

    def _m_step_update_beta(self, beta_all: Tensor) -> Tensor:
        # NOTE: 我自己的实现更快
        WXIS = self._XIS * self._WIS
        for i in range(self._max_iter_inner):
            # grad_o
            p_o = (self._Xo_des @ beta_all).sigmoid()  # ns
            grad = self._Xo_des.T @ (p_o - self._Yo)
            # grad_m
            p_m = (
                self._XIS * beta_all[0] + self._Cm_des @ beta_all[1:]
            ).sigmoid()  # N x nm
            Esig = (p_m * self._WIS).sum(dim=0)
            Esigx = (p_m * WXIS).sum(dim=0)
            grad[0] += (Esigx - self._Ym * self._Xm).sum()
            grad[1:] += self._Cm_des.T @ (Esig - self._Ym)

            # hess_o
            hess = torch.einsum(
                "ij,i,ik->jk", self._Xo_des, p_o * (1 - p_o), self._Xo_des
            )
            p_m2 = p_m * (1 - p_m)
            hess_m_00 = (p_m2 * self._XIS**2 * self._WIS).sum(dim=0).sum()
            hess_m_01 = self._Cm_des.T @ ((p_m2 * WXIS).sum(dim=0))
            hess_m_11 = torch.einsum(
                "ij,i,ik",
                self._Cm_des,
                (p_m2 * self._WIS).sum(axis=0),
                self._Cm_des,
            )
            hess[0, 0] += hess_m_00
            hess[0, 1:] += hess_m_01
            hess[1:, 0] += hess_m_01
            hess[1:, 1:] += hess_m_11
            beta_delta = torch.linalg.solve(hess, grad)
            if self._gem:
                return beta_all - beta_delta

            rdiff = torch.max(
                torch.abs(beta_delta)
                / (torch.abs(beta_all) + self._delta1_inner)
            )
            beta_all = beta_all - beta_delta
            logger_embp.info(
                f"M step Newton-Raphson: iter={i+1} diff={rdiff:.4f}"
            )
            if rdiff < self._delta2_inner:
                break
        else:
            logger_embp.warning(
                f"M step Newton-Raphson (max_iter={self._max_iter_inner})"
                " doesn't converge"
            )

        return beta_all

    def after_iter(self):
        if self._max_nIS > self._min_nIS:
            self._nIS = self._min_nIS + int(
                (self._max_nIS - self._min_nIS)
                * expit(2 * LOGIT_3 * self._iter_i / self._max_iter - LOGIT_3)
            )
            logger_embp.info(
                f"Update Monte Carlo Sampling size to {self._nIS}."
            )
