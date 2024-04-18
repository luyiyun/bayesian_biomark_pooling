import logging

import numpy as np
from scipy.special import expit, log_expit, softmax
from scipy.stats import norm

from scipy.optimize import minimize_scalar

# import pandas as pd
from numpy import ndarray
from numpy.random import Generator

from ..logger import logger_embp
from .base import EM
from .utils import logistic


EPS = 1e-5
LOGIT_3 = 6.9067548


def newton_raphson_beta(
    init_beta: ndarray,
    Xo_des: ndarray,  # ns x (1+S+p)
    Xm_des: ndarray,  # N x nm x (1+S+p)
    Yo: ndarray,
    Ym: ndarray,
    wIS: ndarray,  # N x nm
    lr: float = 1.0,
    max_iter: int = 100,
    delta1: float = 1e-3,
    delta2: float = 1e-4,
):
    # NOTE: 我自己的实现更快
    beta_ = init_beta
    for i in range(max_iter):
        p_o = expit(Xo_des @ beta_)  # ns
        p_m = expit(Xm_des @ beta_)  # N x nm
        mul_o = p_o - Yo  # ns
        mul_m = (p_m - Ym) * wIS  # N x nm
        grad = Xo_des.T @ mul_o + np.einsum("nij,ni->j", Xm_des, mul_m)
        H_o = np.einsum("ij,i,ik->jk", Xo_des, p_o * (1 - p_o), Xo_des)
        H_m = np.einsum(
            "nij,ni,nik->jk", Xm_des, p_m * (1 - p_m) * wIS, Xm_des
        )
        H = H_o + H_m

        try:  # TODO:
            beta_delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError as e:
            np.save("./Xo_des_tmp.npy", Xo_des)
            np.save("./Yo_tmp.npy", Yo)
            # print(np.cov(Xo_des.T))
            # print(H)
            # print(grad)
            # import statsmodels.api as sm

            # model = sm.Logit(Yo, Xo_des)
            # res = model.fit()
            # print(res.summary())
            raise e

        def obj(lr):
            new_beta = beta_ - lr * beta_delta
            p_o = log_expit((2 * Yo - 1) * (Xo_des @ new_beta))  # ns
            p_m = log_expit((2 * Ym - 1) * (Xm_des @ new_beta))  # N x nm
            return -(p_o.sum() + (p_m * wIS).sum())

        res = minimize_scalar(obj, bounds=(0.0, 1.0))
        if not res.success:
            logger_embp.warning(
                "M step optimiztion lr searching fail to converge, "
                f"msg: {res.message}"
            )
        beta_delta *= res.x
        beta_ = beta_ - beta_delta

        rdiff = np.max(np.abs(beta_delta) / (np.abs(beta_) + delta1))
        logger_embp.info(f"M step Newton-Raphson: iter={i+1} diff={rdiff:.4f}")
        if rdiff < delta2:
            break
    else:
        logger_embp.warning(
            f"M step Newton-Raphson (max_iter={max_iter}) doesn't converge"
        )

    return beta_


class BinaryEM(EM):  # TODO: 有错误！！！！
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

    def prepare(self):
        super().prepare()

        C = np.zeros((self._n, self._ns))
        for i in range(self._ns):
            C[self._ind_S[i], i] = 1
        self._Co = C[self._is_o, :]
        self._Cm_ptype = C[self._is_m, :]

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
        Xm_des = [
            self._XIS[:, :, None],
            np.tile(self._Cm_ptype[None, ...], (self._nIS, 1, 1)),
        ]
        if self._Z is not None:
            Xm_des.append(np.tile(self._Zm[None, ...], (self._nIS, 1, 1)))
        Xm_des = np.concatenate(Xm_des, axis=-1)  # N x nm x (1+S+p)
        beta_all = newton_raphson_beta(
            init_beta=beta_all,
            Xo_des=self._Xo_des,
            Xm_des=Xm_des,
            Yo=self._Yo,
            Ym=self._Ym,
            wIS=self._WIS,
            lr=self._lr,
            max_iter=self._max_iter_inner,
            delta1=self._delta1_inner,
            delta2=self._delta2_inner,
        )

        return np.r_[
            mu_x,
            sigma2_x,
            a,
            b,
            sigma2_w,
            beta_all,
        ]

    def after_m_step(self):
        if self._max_nIS > self._min_nIS:
            self._nIS = self._min_nIS + int(
                (self._max_nIS - self._min_nIS)
                * expit(2 * LOGIT_3 * self._iter_i / self._max_iter - LOGIT_3)
            )
            logger_embp.info(
                f"Update Monte Carlo Sampling size to {self._nIS}."
            )
