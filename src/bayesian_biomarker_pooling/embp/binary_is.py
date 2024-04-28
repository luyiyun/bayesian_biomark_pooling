import logging

import numpy as np
from scipy.special import expit, log_expit, softmax
from scipy.stats import norm

# import pandas as pd
from numpy import ndarray
from numpy.random import Generator

from ..logger import logger_embp
from .base import EM
from .utils import logistic


EPS = 1e-5
LOGIT_3 = 6.9067548


class ISBinaryEM(EM):
    def __init__(
        self,
        max_iter: int = 300,
        max_iter_inner: int = 100,
        delta1: float = 1e-3,
        delta1_inner: float = 1e-4,
        delta1_var: float = 1e-2,
        delta2: float = 1e-2,
        delta2_inner: float = 1e-6,
        delta2_var: float = 1e-2,
        pbar: bool = True,
        random_seed: int | None | Generator = None,
        lr: float = 1.0,
        min_nIS: int = 100,
        max_nIS: int = 5000,
        gem: bool = True,
    ) -> None:
        assert max_nIS >= min_nIS
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
        self._lr = lr
        self._nIS = self._min_nIS = min_nIS
        self._max_nIS = max_nIS
        self._gem = gem

    def prepare(self):
        super().prepare()

        C = np.zeros((self._n, self._ns))
        for i in range(self._ns):
            C[self._ind_S[i], i] = 1
        self._Co = C[self._is_o, :]
        self._Cm_des = C[self._is_m, :]
        if self._Z is not None:
            self._Cm_des = np.concatenate([self._Cm_des, self._Zm], axis=1)

        Xo_des = [self._Xo[:, None], self._Co]  # 360 * 5
        if self._Z is not None:
            Xo_des.append(self._Zo)
        self._Xo_des = np.concatenate(Xo_des, axis=-1)
        self._Xm = np.zeros(self._n_m)  # 用于e-step中的newton algorithm

        self._params_ind.update(
            {
                "beta_x": slice(2 + 3 * self._ns, 3 + 3 * self._ns),
                "beta_0": slice(3 + 3 * self._ns, 3 + 4 * self._ns),
                "beta_z": slice(3 + 4 * self._ns, 3 + 4 * self._ns + self._nz),
            }
        )
        self._nparams = 3 + 4 * self._ns + self._nz

    def init(self) -> ndarray:
        """初始化权重"""
        # res = self._seed.random(self._nparams)
        # return res
        params = super().init()
        # beta
        beta = logistic(self._Xo, self._Yo, self._Zo)
        return np.concatenate([params, beta[[0] + [1] * self._ns], beta[2:]])

    def e_step(self, params: ndarray):
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
        a_m_long = a[self._ind_m_inv]
        b_m_long = b[self._ind_m_inv]
        sigma2_w_m_long = sigma2_w[self._ind_m_inv]

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

        for i in range(1, self._max_iter_inner + 1):
            p = expit(self._Xm * beta_x + delta_part)

            xdelta = (
                self._lr
                * (p_mult_m_long * p + x_mult_m_long * self._Xm - const_m_long)
                / (p2_mult_m_long * p * (1 - p) + x_mult_m_long)
            )
            self._Xm -= xdelta

            rdiff = np.max(
                np.abs(xdelta) / (np.abs(self._Xm) + self._delta1_inner)
            )
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

        # 重新计算一次hessian
        # 不要使用multivariate_norm，会导致维数灾难，
        # 因为Xm的每个分量都是独立的，使用单变量的norm会好一些
        p = expit(self._Xm * beta_x + delta_part)
        hess_inv = (sigma2_w * sigma2_x)[self._ind_m_inv] / (
            p2_mult_m_long * p * (1 - p) + x_mult_m_long
        )
        norm_lap = norm(
            loc=self._Xm, scale=np.sqrt(hess_inv) + EPS
        )  # TODO: 这个EPS可能不是必须的

        # 进行IS采样
        self._XIS = norm_lap.rvs(
            size=(self._nIS, self._n_m), random_state=self._seed
        )  # N x n_m

        # 计算对应的(normalized) importance weights
        pIS = log_expit((2 * self._Ym - 1) * (beta_x * self._XIS + delta_part))
        pIS -= 0.5 * (
            (self._Wm - a_m_long - b_m_long * self._XIS) ** 2 / sigma2_w_m_long
            + (self._XIS - mu_x) ** 2 / sigma2_x
        )
        pIS = pIS - norm_lap.logpdf(self._XIS)
        # NOTE: 尽管归一化因子对于求极值没有贡献，但有助于稳定训练
        self._WIS = softmax(pIS, axis=0)

        if logger_embp.level <= logging.INFO:
            Seff = 1 / np.sum(self._WIS**2, axis=0)
            logger_embp.info(
                "Importance effective size "
                + f"is {Seff.mean():.2f}±{Seff.std():.2f}"
            )

        # 计算Xhat和Xhat2
        self._Xhat[self._is_m] = np.sum(self._XIS * self._WIS, axis=0)
        self._Xhat2[self._is_m] = np.sum(self._XIS**2 * self._WIS, axis=0)

    def m_step(self, params: ndarray) -> ndarray:
        vbar = self._Xhat2.mean()
        xbars = np.array([self._Xhat[sind].mean() for sind in self._ind_S])
        vbars = np.array([self._Xhat2[sind].mean() for sind in self._ind_S])
        wxbars = np.array(
            [np.mean(self._W[sind] * self._Xhat[sind]) for sind in self._ind_S]
        )

        # 更新参数：mu_x,sigma2_x,a,b,sigma2_w
        mu_x = self._Xhat.mean()
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
        beta_all = params[self._params_ind["beta_x"].start :]
        beta_all = self._update_beta_all(beta_all)

        return np.r_[
            mu_x,
            sigma2_x,
            a,
            b,
            sigma2_w,
            beta_all,
        ]

    def _update_beta_all(self, beta_all: ndarray) -> ndarray:
        # NOTE: 我自己的实现更快
        WXIS = self._XIS * self._WIS
        for i in range(self._max_iter_inner):
            # grad_o
            p_o = expit(self._Xo_des @ beta_all)  # ns
            grad = self._Xo_des.T @ (p_o - self._Yo)
            # grad_m
            p_m = expit(
                self._XIS * beta_all[0] + self._Cm_des @ beta_all[1:]
            )  # N x nm
            Esig = (p_m * self._WIS).sum(axis=0)
            Esigx = (p_m * WXIS).sum(axis=0)
            grad[0] += (Esigx - self._Ym * self._Xm).sum()
            grad[1:] += self._Cm_des.T @ (Esig - self._Ym)

            # hess_o
            hess = np.einsum(
                "ij,i,ik->jk", self._Xo_des, p_o * (1 - p_o), self._Xo_des
            )
            p_m2 = p_m * (1 - p_m)
            hess_m_00 = (p_m2 * self._XIS**2 * self._WIS).sum(axis=0).sum()
            hess_m_01 = self._Cm_des.T @ (
                (p_m2 * WXIS).sum(axis=0)
            )
            hess_m_11 = np.einsum(
                "ij,i,ik",
                self._Cm_des,
                (p_m2 * self._WIS).sum(axis=0),
                self._Cm_des,
            )
            hess_m = np.block(  # 这种比inplace替换([]+=)更快
                [
                    [hess_m_00, hess_m_01[None, :]],
                    [hess_m_01[:, None], hess_m_11],
                ]
            )
            hess = hess + hess_m
            beta_delta = np.linalg.solve(hess, grad)
            if self._gem:
                return beta_all - beta_delta

            rdiff = np.max(
                np.abs(beta_delta) / (np.abs(beta_all) + self._delta1_inner)
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

    def after_m_step(self):
        if self._max_nIS > self._min_nIS:
            self._nIS = self._min_nIS + int(
                (self._max_nIS - self._min_nIS)
                * expit(2 * LOGIT_3 * self._iter_i / self._max_iter - LOGIT_3)
            )
            logger_embp.info(
                f"Update Monte Carlo Sampling size to {self._nIS}."
            )
