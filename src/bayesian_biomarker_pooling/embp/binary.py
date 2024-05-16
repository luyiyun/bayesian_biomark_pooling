import logging

import numpy as np
from scipy.special import expit, ndtri, log_expit, softmax
from numpy import ndarray
from numpy.random import Generator
from scipy.stats import norm as norm_sc
from scipy.linalg import block_diag as sc_block_diag

from ..logger import logger_embp
from .base import NumpyEM
from .utils import logistic


EPS = 1e-5
LOGIT_3 = 6.9067548


class BinaryEM(NumpyEM):
    def __init__(
        self,
        max_iter: int = 300,
        max_iter_inner: int = 100,
        delta1: float = 1e-3,
        delta1_inner: float = 1e-4,
        delta1_var: float = 1e-2,
        delta2: float = 1e-3,
        delta2_inner: float = 1e-6,
        delta2_var: float = 1e-2,
        pbar: bool = True,
        random_seed: int | None | Generator = None,
        gem: bool = True,
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
        self._Xm = np.zeros(self._n_m)

        C = np.zeros((self._n, self._ns))
        for i in range(self._ns):
            C[self._ind_S[i], i] = 1
        Xo_des = [self._Xo[:, None], C[self._is_o, :]]  # 360 * 5
        Cm_des = [C[self._is_m, :]]
        if self._Z is not None:
            Xo_des.append(self._Zo)
            Cm_des.append(self._Zm)
        self._Xo_des = np.concatenate(Xo_des, axis=-1)
        self._Cm_des = np.concatenate(Cm_des, axis=-1)

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
        params = super().init()
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
            p = expit(self._Xm * beta_x + delta_part)

            xdelta = (
                # self._lr *
                (p_mult_m_long * p + x_mult_m_long * self._Xm - const_m_long)
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

        # 重新计算一次hessian，并计算得到Laplacian Approximation的variance
        p = expit(self._Xm * beta_x + delta_part)
        self._Vm = (sigma2_w * sigma2_x)[self._ind_m_inv] / (
            p2_mult_m_long * p * (1 - p) + x_mult_m_long
        )
        self._Xm_Sigma = np.sqrt(self._Vm)

        self._e_step_update_statistics(
            mu_x=mu_x,
            sigma2_x=sigma2_x,
            beta_x=beta_x,
            a=a,
            b=b,
            sigma2_w=sigma2_w,
            delta_part=delta_part,
        )

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
        # 基于Laplacian Approximation和Importance Sampling的过程会有不同
        beta_all = params[self._params_ind["beta_x"].start :]
        beta_all = self._m_step_update_beta(beta_all)
        return np.r_[
            mu_x,
            sigma2_x,
            a,
            b,
            sigma2_w,
            beta_all,
        ]


class LapBinaryEM(BinaryEM):
    def __init__(
        self,
        max_iter: int = 300,
        max_iter_inner: int = 100,
        delta1: float = 1e-3,
        delta1_inner: float = 1e-4,
        delta1_var: float = 1e-2,
        delta2: float = 1e-3,
        delta2_inner: float = 1e-6,
        delta2_var: float = 1e-2,
        pbar: bool = True,
        random_seed: int | None | Generator = None,
        K: int = 10,
        gem: bool = True,
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
            gem=gem,
        )
        self._K = K  # quasi-monte-carlo
        self._unif_K = (np.arange(1, self._K) / self._K)[:, None]
        self._ppf_sn = ndtri(self._unif_K).squeeze()

    def _e_step_update_statistics(self, **params):
        # 计算Xhat和Xhat2
        self._Xhat[self._is_m] = self._Xm
        self._Xhat2[self._is_m] = self._Xm**2 + self._Vm

    def _m_step_update_beta(self, beta_all: ndarray):
        # NOTE: 我自己的实现更快
        # 使用这个替代ppf函数，更快
        h = self._ppf_sn[:, None] * self._Xm_Sigma + self._Xm
        for i in range(self._max_iter_inner):
            # 计算grad_o
            p_o = expit(self._Xo_des @ beta_all)  # no
            grad = self._Xo_des.T @ (p_o - self._Yo)
            # 计算grad_m
            sigma = expit(beta_all[0] * h + self._Cm_des @ beta_all[1:])
            Esig = sigma.mean(axis=0)
            Esigx = (sigma * h).mean(axis=0)
            grad_m_betax = (Esigx - self._Ym * self._Xm).sum()
            grad_m_other = self._Cm_des.T @ (Esig - self._Ym)
            grad[0] += grad_m_betax  # inplace更快
            grad[1:] += grad_m_other

            # 计算hess_o
            hess = np.einsum(
                "ij,i,ik->jk", self._Xo_des, p_o * (1 - p_o), self._Xo_des
            )
            # 计算hess_m
            sigma2 = sigma * (1 - sigma)
            hess_m_00 = (sigma2 * h**2).mean(axis=0).sum()
            hess_m_01 = self._Cm_des.T @ (sigma2 * h).mean(axis=0)
            hess_m_11 = np.einsum(
                "ij,i,ik", self._Cm_des, sigma2.mean(axis=0), self._Cm_des
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

    def v_joint(self, params: ndarray) -> ndarray:
        mu_x, sigma2_x, a, b, sigma2_w = (
            params[self._params_ind[k]]
            for k in [
                "mu_x",
                "sigma2_x",
                "a",
                "b",
                "sigma2_w",
            ]
        )
        beta_all = params[self._params_ind["beta_x"].start :]
        mu_x, sigma2_x = mu_x[0], sigma2_x[0]  # array->item

        self.e_step(params)

        xbar = self._Xhat.mean()
        vbar = self._Xhat2.mean()
        xbars = np.array([self._Xhat[sind].mean() for sind in self._ind_S])
        vbars = np.array([self._Xhat2[sind].mean() for sind in self._ind_S])
        wxbars = np.array(
            [np.mean(self._W[sind] * self._Xhat[sind]) for sind in self._ind_S]
        )

        x12 = (xbar - mu_x) / sigma2_x
        V1 = self._n * np.array(
            [
                [1 / sigma2_x, x12],
                [x12, 0.5 * (vbar - mu_x**2) / sigma2_x],
            ]
        )

        temp_mul = self._n_s / sigma2_w
        A = np.diag(temp_mul)
        B = np.diag(temp_mul * xbars)
        C = np.diag(temp_mul * (self._wbar_s - a - b * xbars))
        D = np.diag(temp_mul * vbars)
        E = np.diag(temp_mul * (wxbars - a * xbars - b * vbars))
        F = np.diag(
            0.5
            * temp_mul
            * (
                self._wwbar_s
                + a**2
                + b**2 * vbars
                - 2 * a * self._wbar_s
                - 2 * b * wxbars
                + 2 * a * b * xbars
            )
        )
        V2 = np.block([[A, B, C], [B, D, E], [C, E, F]])

        # 使用这个替代ppf函数，更快
        h = self._ppf_sn[:, None] * self._Xm_Sigma + self._Xm
        p2 = np.zeros(self._n)
        po = expit(self._Xo_des @ beta_all)
        p2[self._ind_o] = po * (1 - po)
        p2x = p2 * self._X
        p2xx = p2x * self._X

        pm_ = expit(beta_all[0] * h + self._Cm_des @ beta_all[1:])
        pm2_ = pm_ * (1 - pm_)
        p2[self._ind_m] = pm2_.mean(axis=0)
        p2x[self._ind_m] = (pm2_ * h).mean(axis=0)
        p2xx[self._ind_m] = (pm2_ * h * h).mean(axis=0)

        K = np.array([[p2xx.sum()]])
        A = np.array([[p2x[ind].sum() for ind in self._ind_S]])
        D = np.diag([p2[ind].sum() for ind in self._ind_S])
        V3 = [[K, A], [A.T, D]]
        if self._Z is not None:
            B = (self._Z.T @ p2x)[None, :]
            p2z = p2[:, None] * self._Z
            E = np.stack(
                [p2z[ind].sum(axis=0) for ind in self._ind_S],
                axis=0,
            )
            G = np.einsum("ij,ik,i->jk", self._Z, self._Z, p2)
            V3[0].insert(2, B)
            V3[1].insert(2, E)
            V3.insert(2, [B.T, E.T, G])
        V3 = np.block(V3)

        return sc_block_diag(
            np.linalg.inv(V1), np.linalg.inv(V2), np.linalg.inv(V3)
        )


class ISBinaryEM(BinaryEM):
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
            gem=gem,
        )
        self._lr = lr
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

        norm_lap = norm_sc(
            loc=self._Xm, scale=self._Xm_Sigma + EPS
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

        # 计算Xhat和Xhat2, 并讲self._Xm更新为IS计算的后验均值
        self._Xhat[self._is_m] = self._Xm = np.sum(
            self._XIS * self._WIS, axis=0
        )
        self._Xhat2[self._is_m] = np.sum(self._XIS**2 * self._WIS, axis=0)

    def _m_step_update_beta(self, beta_all: ndarray) -> ndarray:
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
            hess_m_01 = self._Cm_des.T @ ((p_m2 * WXIS).sum(axis=0))
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

    def after_iter(self):
        if self._max_nIS > self._min_nIS:
            self._nIS = self._min_nIS + int(
                (self._max_nIS - self._min_nIS)
                * expit(2 * LOGIT_3 * self._iter_i / self._max_iter - LOGIT_3)
            )
            logger_embp.info(
                f"Update Monte Carlo Sampling size to {self._nIS}."
            )
