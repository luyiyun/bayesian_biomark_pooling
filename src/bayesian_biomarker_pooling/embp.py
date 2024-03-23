import logging
from typing import Literal
from copy import deepcopy

import pandas as pd
import numpy as np
from scipy.special import expit, log_expit, softmax
from scipy.stats import norm
from scipy.linalg import block_diag
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# import pandas as pd
from numpy import ndarray
from numpy.random import Generator

from .base import BiomarkerPoolBase
from .logger import logger_embp


EPS = 1e-7


def ols(x_des, y) -> np.ndarray:
    return np.linalg.inv(x_des.T @ x_des) @ x_des.T @ y


def logistic(
    Xdes,
    y,
    lr: float = 1.0,
    max_iter: int = 100,
    delta1: float = 1e-3,
    delta2: float = 1e-4,
):
    beta = np.zeros(Xdes.shape[1])

    for i in range(max_iter):
        p = expit(Xdes @ beta)
        grad = Xdes.T @ (p - y)
        hessian = (Xdes.T * p * (1 - p)) @ Xdes
        delta = lr * np.linalg.inv(hessian) @ grad
        beta -= delta

        rdiff = np.max(np.abs(delta) / (np.abs(beta) + delta1))
        logger_embp.info(f"Init Newton-Raphson: iter={i+1} diff={rdiff:.4f}")
        if rdiff < delta2:
            break
    else:
        logger_embp.warning(
            f"Init Newton-Raphson (max_iter={max_iter}) doesn't converge"
        )

    return beta


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
    beta_ = init_beta
    for i in range(max_iter):
        p_o = expit(Xo_des @ beta_)  # ns
        p_m = expit(Xm_des @ beta_)  # N x nm
        mul_o = p_o - Yo  # ns
        mul_m = (p_m - Ym) * wIS  # N x nm
        grad = Xo_des.T @ mul_o + np.sum(
            Xm_des * mul_m[..., None], axis=(0, 1)
        )  # (1+S+p)

        H_o = (Xo_des.T * p_o * (1 - p_o)) @ Xo_des
        H_m = np.sum(
            (Xm_des * (wIS * p_m * (1 - p_m))[..., None]).swapaxes(1, 2)
            @ Xm_des,
            axis=0,
        )
        H = H_o + H_m

        beta_delta = lr * np.linalg.inv(H) @ grad
        beta_ -= beta_delta

        rdiff = np.max(np.abs(beta_delta) / (np.abs(beta_) + delta1))
        logger_embp.info(f"M step Newton-Raphson: iter={i+1} diff={rdiff:.4f}")
        if rdiff < delta2:
            break
    else:
        logger_embp.warning(
            f"M step Newton-Raphson (max_iter={max_iter}) doesn't converge"
        )

    return beta_


class EM:

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
        init_method: Literal["rand", "reference"] = "reference",
        pbar: bool = True,
        random_seed: int | None | Generator = None,
    ) -> None:
        assert init_method in ["rand", "reference"]

        self._max_iter = max_iter
        self._max_iter_inner = max_iter_inner
        self._delta1 = delta1
        self._delta1_inner = delta1_inner
        self._delta2 = delta2
        self._delta2_inner = delta2_inner
        self._delta1_var = delta1_var
        self._delta2_var = delta2_var
        self._init_method = init_method
        self._pbar = pbar
        # 如果是Generator，则default_rng会直接返回它自身
        self._seed = np.random.default_rng(random_seed)

    def register_data(
        self,
        X: ndarray,
        S: ndarray,
        W: ndarray,
        Y: ndarray,
        Z: ndarray | None = None,
    ):
        self._X = X
        self._S = S
        self._W = W
        self._Y = Y
        self._Z = Z

    def prepare(self):
        # 准备后续步骤中会用到的array，预先计算，节省效率
        self._n = self._Y.shape[0]
        self._studies, self._ind_inv = np.unique(self._S, return_inverse=True)
        self._is_m = pd.isnull(self._X)
        self._is_o = ~self._is_m
        self._ns = len(self._studies)
        self._n_o = self._is_o.sum()
        self._n_m = self._is_m.sum()
        self._nz = 0 if self._Z is None else self._Z.shpae[1]

        self._Xo = self._X[self._is_o]
        self._Yo = self._Y[self._is_o]
        self._Wo = self._W[self._is_o]
        self._Xm = self._X[self._is_m]
        self._Ym = self._Y[self._is_m]
        self._Wm = self._W[self._is_m]
        if self._Z is not None:
            self._Zo = self._Z[self._is_o, :]
            self._Zm = self._Z[self._is_m, :]

        self._ind_S = [np.nonzero(self._S == s)[0] for s in self._studies]
        self._ind_Sm = [
            np.nonzero((self._S == s) & self._is_m)[0] for s in self._studies
        ]
        self._ind_So = [
            np.nonzero((self._S == s) & self._is_o)[0] for s in self._studies
        ]
        self._ind_m_inv = self._ind_inv[self._is_m]

        self._n_s = np.array([len(indi) for indi in self._ind_S])
        self._n_ms = np.array([len(indi) for indi in self._ind_Sm])
        # self._ind_Sm_inv = [self._ind_inv[is_i] for is_i in self._ind_Sm]

        self._wbar_s = np.array([np.mean(self._W[ind]) for ind in self._ind_S])
        self._wwbar_s = np.array(
            [np.mean(self._W[ind] ** 2) for ind in self._ind_S]
        )

        self._Xhat = np.copy(self._X)
        self._Xhat2 = self._Xhat**2

    def init(self) -> pd.Series:
        """初始化参数

        这里仅初始化mu_x,sigma2_x,a,b,sigma2_w，其他和outcome相关的参数需要在子类
        中初始化

        Returns:
            dict: 参数组成的dict
        """
        if self._init_method == "reference":
            mu_x = self._Xo.mean()
            sigma2_x = np.var(self._Xo, ddof=1)

            a, b, sigma2_w = [], [], []
            for ind_so_i in self._ind_So:
                if len(ind_so_i) == 0:
                    a.append(0)
                    b.append(0)
                    sigma2_w.append(1)
                    continue

                Xi, Wi = self._X[ind_so_i], self._W[ind_so_i]
                Xi_des = np.stack([np.ones(Xi.shape[0]), Xi], axis=1)
                abi = ols(Xi_des, Wi)
                sigma2_ws_i = np.mean((Wi - Xi_des @ abi) ** 2)
                a.append(abi[0])
                b.append(abi[1])
                sigma2_w.append(sigma2_ws_i)

            return pd.Series(
                [mu_x, sigma2_x] + a + b + sigma2_w,
                index=["mu_x", "sigma2_x"]
                + ["a"] * self._ns
                + ["b"] * self._ns
                + ["sigma2_w"] * self._ns,
            )
        else:
            arr = self._seed.normal(size=self._ns * 3 + 2)
            arr[1] = np.exp(arr[1])
            arr[-self._ns :] = np.exp(arr[-self._ns :])
            return pd.Series(
                arr,
                index=["mu_x", "sigma2_x"]
                + ["a"] * self._ns
                + ["b"] * self._ns
                + ["sigma2_w"] * self._ns,
            )

    def e_step(self, params: pd.Series):
        """Expectation step

        从EM算法的定义上，是计算log joint likelihood的后验期望，也就是Q function。
        但是，在code中一般不是计算这个，而是计算Q function中关于后验期望的部分，
        以便于后面的m step。
        """
        raise NotImplementedError

    def m_step(self, params: pd.Series) -> pd.Series:
        raise NotImplementedError

    def run(self, init_params: pd.Series | None = None):
        self.prepare()

        params = self.init() if init_params is None else init_params
        self.params_hist_ = [params]
        with logging_redirect_tqdm(loggers=[logger_embp]):
            for iter_i in tqdm(
                range(1, self._max_iter + 1),
                desc="EM: ",
                disable=not self._pbar,
            ):

                self.e_step(params)
                params_new = self.m_step(params)
                # diff = np.max(np.abs(params - params_new))
                rdiff = np.max(
                    np.abs(params - params_new)
                    / (np.abs(params) + self._delta1)
                )
                logger_embp.info(
                    f"EM iteration {iter_i}: "
                    f"relative difference is {rdiff: .4f}"
                )
                params = params_new  # 更新
                self.params_hist_.append(params)
                if rdiff < self._delta2:
                    self.iter_convergence_ = iter_i
                    break
            else:
                logger_embp.warning(
                    f"EM iteration (max_iter={self._max_iter}) "
                    "doesn't converge"
                )

        self.params_ = params
        self.params_hist_ = pd.concat(self.params_hist_, axis=1).T

    def v_joint(self, params: pd.Series) -> ndarray:
        raise NotImplementedError

    def estimate_variance(self) -> ndarray:
        n_params = self.params_.shape[0]

        ind_sigma2 = np.nonzero(
            self.params_.index.map(lambda x: x.startswith("sigma2"))
        )[0]
        params_w_log = self.params_.copy()
        params_w_log.iloc[ind_sigma2] = np.log(params_w_log.iloc[ind_sigma2])

        rind_uncovg = list(range(n_params))
        R = []
        with logging_redirect_tqdm(loggers=[logger_embp]):
            for t in tqdm(
                range(self.params_hist_.shape[0]),
                desc="Estimate Variance: ",
                disable=not self._pbar,
            ):
                params_t = self.params_hist_.iloc[t, :]
                Rt = np.zeros((n_params, n_params)) if t == 0 else R[-1].copy()
                for j in rind_uncovg:
                    inpt = self.params_.copy()
                    x = inpt.iloc[j] = params_t.iloc[j]
                    if j in ind_sigma2:
                        x = np.log(x)
                    # 计算差值比来作为导数的估计
                    dx = x - params_w_log.iloc[j]
                    if (
                        dx == 0
                    ):  # 如果dx=0了，就用上一个结果  TODO: 方差可能还没有收敛
                        continue

                    self.e_step(inpt)
                    oupt = self.m_step(inpt)
                    # 修改sigma2为log尺度
                    oupt.iloc[ind_sigma2] = np.log(oupt.iloc[ind_sigma2])

                    Rt[j, :] = (oupt.values - params_w_log.values) / dx

                # 看一下有哪些行完成了收敛
                if t > 0:
                    rdiff = np.max(
                        np.abs(Rt - R[-1])
                        / (np.abs(R[-1]) + self._delta1_var),
                        axis=1,
                    )
                    new_rind_uncovg = np.nonzero(rdiff >= self._delta2_var)[0]
                    if len(new_rind_uncovg) < len(rind_uncovg):
                        logger_embp.info(
                            "unfinished row ind:" + str(rind_uncovg)
                        )
                    rind_uncovg = new_rind_uncovg

                R.append(Rt)
                if len(rind_uncovg) == 0:
                    break
            else:
                logger_embp.warn("estimate variance does not converge.")

        self._R = np.stack(R, axis=0)
        DM = R[-1]

        v_joint = self.v_joint(self.params_)
        self.params_cov_ = v_joint + v_joint @ DM @ np.linalg.inv(
            np.diag(np.ones(n_params)) - DM
        )
        # TODO: 会出现一些方差<0，可能源于
        params_var = np.diag(self.params_cov_)
        # if np.any(params_var < 0.0):
        #     print(self.params_.index.values[params_var < 0.0])
        #     dv = params_var - np.diag(v_joint)
        #     import ipdb; ipdb.set_trace()
        return params_var


class ContinueEM(EM):

    def prepare(self):
        super().prepare()

        self._ybar_s = np.array([self._Y[ind].mean() for ind in self._ind_S])
        self._yybar_s = np.array(
            [(self._Y[ind] ** 2).mean() for ind in self._ind_S]
        )

        if self._Z is not None:
            self._zzbar_s = []
            self._zbar_s = []
            self._yzbar_s = []
            for ind in self._ind_S:
                Zs = self._Z[ind, :]
                self._zzbar_s.append(Zs.T @ Zs / ind.shape[0])
                self._zbar_s.append(Zs.mean(axis=0))
                self._yzbar_s.append((Zs * self._Y[ind]).mean())
            self._zzbar_s = np.stack(self._zzbar_s, axis=0)
            self._zbar_s = np.stack(self._zbar_s)
            self._yzbar_s = np.stack(self._yzbar_s)
        else:
            self._zzbar_s = self._zbar_s = self._yzbar_s = 0

    def init(self) -> pd.Series:
        params = super().init()

        if self._init_method == "reference":
            Xo_des = [np.ones((self._Xo.shape[0], 1)), self._Xo[:, None]]
            if self._Z is not None:
                Xo_des.append(self._Zo)
            Xo_des = np.concatenate(Xo_des, axis=1)
            beta = ols(Xo_des, self._Yo)
            sigma2_ys = np.mean((self._Yo - Xo_des @ beta) ** 2)

            return pd.concat(
                [
                    params,
                    pd.Series(
                        [beta[1]]
                        + [beta[0]] * self._ns
                        + beta[2:].tolist()
                        + [sigma2_ys] * self._ns,
                        index=["beta_x"]
                        + ["beta_0"] * self._ns
                        + ["beta_z"] * self._nz
                        + ["sigma2_y"] * self._ns,
                    ),
                ]
            )
        else:
            beta = self._seed.normal(size=self._ns * 2 + self._nz + 1)
            beta[1 : (1 + self._ns)] = np.exp(beta[1 : (1 + self._ns)])
            beta[-self._ns :] = np.exp(beta[-self._ns :])
            return pd.concat(
                [
                    params,
                    pd.Series(
                        beta,
                        index=["beta_x"]
                        + ["beta_0"] * self._ns
                        + ["beta_z"] * self._nz
                        + ["sigma2_y"] * self._ns,
                    ),
                ]
            )

    def e_step(self, params: pd.Series):
        mu_x = params["mu_x"]
        sigma2_x = params["sigma2_x"]
        a = params["a"].values
        b = params["b"].values
        sigma2_w = params["sigma2_w"].values
        beta_x = params["beta_x"]
        beta_z = params["beta_z"].values if self._Z is not None else 0.0
        beta_0 = params["beta_0"].values
        sigma2_y = params["sigma2_y"].values

        # if np.any(sigma2_y == 0) or np.any(sigma2_w ==0) or (sigma2_x ==0):
        #     import ipdb; ipdb.set_trace()
        self._sigma2 = 1 / (
            beta_x**2 / sigma2_y + b**2 / sigma2_w + 1 / sigma2_x
        )  # 最后在迭代计算sigma2_y的时候还会用到
        z_m_part = 0.0 if self._Z is None else self._Zm @ beta_z
        beta_0_m_long = beta_0[self._ind_m_inv]
        sigma2_y_m_long = sigma2_y[self._ind_m_inv]
        a_m_long = a[self._ind_m_inv]
        b_m_long = b[self._ind_m_inv]
        sigma2_w_m_long = sigma2_w[self._ind_m_inv]
        sigma2_m_long = self._sigma2[self._ind_m_inv]

        xhat_m = (
            (self._Ym - beta_0_m_long - z_m_part) * beta_x / sigma2_y_m_long
            + (self._Wm - a_m_long) * b_m_long / sigma2_w_m_long
            + mu_x / sigma2_x
        ) * sigma2_m_long

        self._Xhat[self._is_m] = xhat_m
        self._Xhat2[self._is_m] = xhat_m**2 + sigma2_m_long

    def m_step(self, params: pd.Series) -> pd.Series:
        vbar = self._Xhat2.mean()
        wxbar_s = np.array(
            [np.mean(self._W[ind] * self._Xhat[ind]) for ind in self._ind_S]
        )
        vbar_s = np.array([np.mean(self._Xhat2[ind]) for ind in self._ind_S])
        xbar_s = np.array([np.mean(self._Xhat[ind]) for ind in self._ind_S])

        xybar_s = np.array(
            [np.mean(self._Xhat[ind] * self._Y[ind]) for ind in self._ind_S]
        )
        if self._Z is not None:
            # xzbar = np.mean(self._Xhat[:, None] * self._Z, axis=0)
            xzbar_s = np.stack(
                [
                    np.mean(self._Xhat[ind, None] * self._Z[ind, :], axis=0)
                    for ind in self._ind_S
                ],
                axis=0,
            )

        # 3. M step，更新参数值
        mu_x = np.mean(self._Xhat)
        sigma2_x = vbar - mu_x**2
        b = (wxbar_s - self._wbar_s * xbar_s) / (vbar_s - xbar_s**2)
        a = self._wbar_s - b * xbar_s
        sigma2_w = (
            self._wwbar_s
            + a**2
            + b**2 * vbar_s
            - 2 * (a * self._wbar_s + b * wxbar_s - a * b * xbar_s)
        )

        # 迭代更新beta值
        beta_all = params.loc["beta_x":].values
        # beta_x = params["beta_x"]  # beta_x作为第一个计算值，是用不到的
        beta_0 = params["beta_0"].values
        sigma2_y = params["sigma2_y"].values
        if self._Z is not None:
            beta_z = params["beta_z"].values
        for i in range(self._max_iter_inner):
            # 关于z的一些项
            if self._Z is not None:
                xzd = xzbar_s @ beta_z
                zd = self._zbar_s @ beta_z
                dzzd = np.sum(
                    beta_z * self._zzbar_s * beta_z[:, None], axis=(1, 2)
                )
                yzd = self._yzbar_s @ beta_z
                xzd = xzbar_s @ beta_z
            else:
                xzd = yzd = dzzd = zd = xzd = 0.0
            # beta_x
            beta_x_new = (
                self._n_s / sigma2_y * (xybar_s - beta_0 * xbar_s - xzd)
            ).sum() / (self._n_s * vbar_s / sigma2_y).sum()
            # beta_0
            beta_0_new = self._ybar_s - zd - beta_x_new * xbar_s
            # sigma2_y
            sigma2_y_new = (
                self._yybar_s
                + beta_0_new**2
                + beta_x_new**2 * vbar_s
                + dzzd
                - 2 * beta_0_new * self._ybar_s
                - 2 * beta_x_new * xybar_s
                - 2 * yzd
                + 2 * beta_0_new * beta_x_new * xbar_s
                + 2 * beta_0_new * zd
                + 2 * beta_x_new * xzd
            )
            # beta_z
            if self._Z is not None:
                beta_z_new = np.linalg.inv(
                    np.sum(
                        self._n_s[:, None, None]
                        * sigma2_y[:, None, None]
                        * self._zzbar_s,
                        axis=0,
                    )
                ) @ (
                    self._n_s
                    / sigma2_y
                    * (
                        self._yzbar_s
                        - beta_0_new[:, None] * self._zbar_s
                        - beta_x_new * xzbar_s
                    )
                )

            # calculate the relative difference
            beta_all_new = np.r_[
                beta_x_new,
                beta_0_new,
                [] if self._Z is None else beta_z_new,
                sigma2_y_new,
            ]
            rdiff = np.max(
                np.abs(beta_all_new - beta_all)
                / (np.abs(beta_all) + self._delta1_inner)
            )
            logger_embp.info(
                f"Inner iteration {i+1}: "
                f"relative difference is {rdiff: .4f}"
            )

            # update parameters
            beta_all = beta_all_new
            # beta_x = beta_x_new
            beta_0 = beta_0_new
            sigma2_y = sigma2_y_new
            if self._Z is not None:
                beta_z = beta_z_new

            if rdiff < self._delta2_inner:
                logger_embp.info(f"Inner iteration stop, stop iter: {i+1}")
                break

        else:
            logger_embp.warn("Inner iteration does not converge")

        return pd.Series(
            np.r_[
                mu_x,
                sigma2_x,
                a,
                b,
                sigma2_w,
                beta_all,
            ],
            index=params.index,
        )

    def iter_calc_params(
        self,
        beta_x: float,
        beta_0: ndarray,
        sigma2_y: ndarray,
        beta_z: ndarray | None,
    ) -> tuple:

        for iter_i in range(1, self._max_iter_inner + 1):
            # 1. update beta_x
            beta_0_long = beta_0[self._ind_inv]
            sigma2_y_long = sigma2_y[self._ind_inv]
            z_part = self._Z @ beta_z if self._Z is not None else 0
            beta_x_new = (
                (self._Y - beta_0_long - z_part) * self._Xhat / sigma2_y_long
            ).mean() / (self._Xhat2 / sigma2_y_long).mean()
            # 2. update beta_0
            resid_beta0 = self._Y - z_part - beta_x_new * self._Xhat
            beta_0_new = np.array(
                [resid_beta0[indi].mean() for indi in self._ind_S]
            )
            # 3. update sigma2_y
            beta_0_long = beta_0_new[self._ind_inv]
            resid_sigma_2 = (
                self._Y - beta_0_long - beta_x_new * self._Xhat - z_part
            ) ** 2
            sigma2_y_new = np.array(
                [resid_sigma_2[ind_si].mean() for ind_si in self._ind_S]
            )
            sigma2_y_new += (
                self._n_ms * beta_x_new**2 * self._sigma2 / self._n_s
            )
            # 4. update beta_z
            if self._Z is not None:
                sigma2_y_long = sigma2_y_new[self._ind_inv]
                resid_z = (
                    self._Y - beta_0_long - beta_x_new * self._Xhat
                ) / sigma2_y_long
                beta_z_new = (
                    np.linalg.inv((self._Z.T * sigma2_y_long) @ self._Z)
                    @ self._Z.T
                    @ resid_z
                )

            # calc diff
            diff = np.r_[
                beta_x_new - beta_x,
                beta_0_new - beta_0,
                sigma2_y_new - sigma2_y,
            ]
            if self._Z is not None:
                diff = np.r_[diff, beta_z_new - beta_z]
            diff = np.max(np.abs(diff))

            logger_embp.info(
                f"Inner iteration {iter_i}: difference is {diff: .4f}"
            )

            beta_x = beta_x_new
            beta_0 = beta_0_new
            sigma2_y = sigma2_y_new
            if self._Z is not None:
                beta_z = beta_z_new

            if diff < self._thre_inner:
                logger_embp.info(f"Inner iteration stop, stop iter: {iter_i}")
                break

        return beta_x, beta_0, sigma2_y, beta_z

    def v_joint(self, params: pd.Series) -> ndarray:
        mu_x = params["mu_x"]
        sigma2_x = params["sigma2_x"]
        a = params["a"].values
        b = params["b"].values
        sigma2_w = params["sigma2_w"].values
        beta_x = params["beta_x"]
        beta_z = params["beta_z"].values if self._Z is not None else 0.0
        beta_0 = params["beta_0"].values
        sigma2_y = params["sigma2_y"].values

        self.e_step(params)

        xbar = self._Xhat.mean()
        vbar = self._Xhat2.mean()
        wxbar_s = np.array(
            [np.mean(self._W[ind] * self._Xhat[ind]) for ind in self._ind_S]
        )
        vbar_s = np.array([np.mean(self._Xhat2[ind]) for ind in self._ind_S])
        xbar_s = np.array([np.mean(self._Xhat[ind]) for ind in self._ind_S])

        xybar_s = np.array(
            [np.mean(self._Xhat[ind] * self._Y[ind]) for ind in self._ind_S]
        )
        if self._Z is not None:
            xzbar = np.mean(self._Xhat[:, None] * self._Z, axis=0)
            xzbar_s = np.stack(
                [
                    np.mean(self._Xhat[ind, None] * self._Z[ind, :], axis=0)
                    for ind in self._ind_S
                ],
                axis=0,
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
        B = np.diag(temp_mul * xbar_s)
        C = np.diag(temp_mul * (self._wbar_s - a - b * xbar_s))
        D = np.diag(temp_mul * vbar_s)
        E = np.diag(temp_mul * (wxbar_s - a * xbar_s - b * vbar_s))
        F = np.diag(
            0.5
            * temp_mul
            * (
                self._wwbar_s
                + a**2
                + b**2 * vbar_s
                - 2 * a * self._wbar_s
                - 2 * b * wxbar_s
                + 2 * a * b * xbar_s
            )
        )
        V2 = np.block([[A, B, C], [B, D, E], [C, E, F]])

        sigma2_y_long = sigma2_y[self._ind_inv]
        temp_mul = self._n_s / sigma2_y
        if self._Z is not None:
            B = np.sum(self._n_s[None, :] * xzbar_s, axis=0, keepdims=True)
            E = temp_mul[None, :] * self._zbar_s
            G = (self._Z.T * sigma2_y_long) @ self._Z
            H = temp_mul * (
                self._yzbar_s.T
                - beta_0 * self._zbar_s.T
                - beta_x * xzbar_s.T
                - (self._zzbar_s @ beta_z).T
            )

            dz2 = np.sum(
                beta_z * beta_z[:, None] * self._zzbar_s,
                axis=(1, 2),
            )
            J_zpart = (
                dz2
                - 2 * self._yzbar_s @ beta_z
                + 2 * beta_0 * self._zbar_s @ beta_z
                + 2 * xzbar_s * beta_z * beta_x
            )
            C_zpart = -xzbar @ beta_z
            F_zpart = -self._zbar_s @ beta_z
        else:
            J_zpart = 0.0
            C_zpart = 0.0
            F_zpart = 0.0
        K = np.array([[np.sum(temp_mul * vbar_s)]])
        A = (temp_mul * xbar_s)[None, :]
        C = (
            temp_mul * (xybar_s - beta_0 * xbar_s - beta_x * vbar_s + C_zpart)
        )[None, :]
        D = np.diag(temp_mul)
        F = np.diag(
            temp_mul * (self._ybar_s - beta_0 - beta_x * xbar_s + F_zpart)
        )
        J = np.diag(
            0.5
            * temp_mul
            * (
                self._yybar_s
                + beta_0**2
                + beta_x**2 * vbar_s
                - 2 * beta_0 * self._ybar_s
                - 2 * beta_x * xybar_s
                + 2 * beta_0 * beta_x * xbar_s
                + J_zpart
            )
        )
        V3 = [[K, A, C], [A.T, D, F], [C.T, F.T, J]]
        if self._Z is not None:
            V3[0].insert(2, B)
            V3[1].insert(2, E)
            V3[2].insert(2, H.T)
            V3.insert(2, [B.T, E.T, G, H])
        V3 = np.block(V3)

        return block_diag(
            np.linalg.inv(V1), np.linalg.inv(V2), np.linalg.inv(V3)
        )


class BinaryEM(EM):

    def __init__(
        self,
        max_iter: int = 500,
        max_iter_inner: int = 100,
        delta1: float = 1e-3,
        delta1_inner: float = 1e-4,
        delta1_var: float = 1e-2,
        delta2: float = 1e-4,
        delta2_inner: float = 1e-6,
        delta2_var: float = 1e-2,
        pbar: bool = True,
        lr: float = 1.0,
        n_importance_sampling: int = 1000,
        seed: int | None | Generator = None,
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
            random_seed=seed,
        )
        self._lr = lr
        self._nIS = n_importance_sampling

    def prepare(self):
        super().prepare()

        C = np.zeros((self._n, self._ns))
        for i in range(self._ns):
            C[self._ind_S[i], i] = 1
        self._Co = C[self._is_o, :]
        self._Cm = np.tile(C[None, self._is_m, :], (self._nIS, 1, 1))

        Xo_des = [self._Xo[:, None], self._Co]  # 360 * 5
        if self._Z is not None:
            Xo_des.append(self._Zo)
        self._Xo_des = np.concatenate(Xo_des, axis=-1)

    def init(self) -> pd.Series:
        """初始化权重"""
        params = super().init()

        # beta
        Xo_des = [np.ones((self._Xo.shape[0], 1)), self._Xo[:, None]]
        if self._Z is not None:
            Xo_des.append(self._Zo)
        Xo_des = np.concatenate(Xo_des, axis=1)
        beta = logistic(
            Xo_des,
            self._Yo,
            lr=self._lr,
            max_iter=self._max_iter_inner,
            delta1=self._delta1_inner,
            delta2=self._delta2_inner,
        )

        return pd.concat(
            [
                params,
                pd.Series(
                    [beta[1]] + [beta[0]] * self._ns + beta[2:].tolist(),
                    index=["beta_x"]
                    + ["beta_0"] * self._ns
                    + ["beta_z"] * (len(beta) - 2),
                ),
            ]
        )

    def e_step(self, params: pd.Series):
        mu_x = params["mu_x"]
        sigma2_x = params["sigma2_x"]
        a = params["a"].values
        b = params["b"].values
        sigma2_w = params["sigma2_w"].values
        beta_x = params["beta_x"]
        beta_z = params["beta_z"].values if self._Z is not None else 0.0
        beta_0 = params["beta_0"].values

        beta_0_m_long = beta_0[self._ind_m_inv]
        a_m_long = a[self._ind_m_inv]
        b_m_long = b[self._ind_m_inv]
        sigma2_w_m_long = sigma2_w[self._ind_m_inv]

        # 使用newton-raphson方法得到Laplacian approximation
        Xm = 0  # init
        b_m_long_2 = b_m_long**2
        beta_x_2 = beta_x**2
        grad_const = (
            -beta_x * self._Ym
            + b_m_long * (a_m_long - self._Wm) / sigma2_w_m_long
            - mu_x / sigma2_x
        )
        grad_mul = b_m_long_2 / sigma2_w_m_long + 1 / sigma2_x

        Z_part_m = 0.0 if self._Z is None else self._Zm @ beta_z
        delta_part = beta_0_m_long + Z_part_m
        for i in range(1, self._max_iter_inner + 1):
            p = expit(Xm * beta_x + delta_part)
            grad = beta_x * p + grad_mul * Xm + grad_const
            hessian = beta_x_2 * p * (1 - p) + grad_mul

            xdelta = self._lr * grad / hessian
            Xm -= xdelta

            rdiff = np.max(np.abs(xdelta) / (np.abs(Xm) + self._delta1_inner))
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

        # 重新计算一次Hessian
        p = expit(Xm * beta_x + delta_part)
        hessian = beta_x_2 * p * (1 - p) + grad_mul

        # 不要使用multivariate_norm，会导致维数灾难，
        # 因为Xm的每个分量都是独立的，使用单变量的norm会好一些
        norm_lap = norm(loc=Xm, scale=1 / np.sqrt(hessian))

        # 进行IS采样
        self._XIS = norm_lap.rvs(
            size=(self._nIS, self._n_m), random_state=self._seed
        )  # N x n_m

        # 计算对应的(normalized) importance weights
        pIS = log_expit(
            (2 * self._Ym - 1)
            * (beta_0_m_long + beta_x * self._XIS + Z_part_m)
        )
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

    def m_step(self, params: pd.Series) -> dict:
        beta_x = params["beta_x"]
        beta_z = params["beta_z"].values if self._Z is not None else []
        beta_0 = params["beta_0"].values

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
        beta_ = np.r_[beta_x, beta_0, beta_z]  # 1Sp = 1+S+p
        Xm_des = [self._XIS[:, :, None], self._Cm]
        if self._Z is not None:
            Xm_des.append(self._Zm[None, ...])
        Xm_des = np.concatenate(Xm_des, axis=-1)  # N x nm x (1+S+p)

        beta_ = newton_raphson_beta(
            init_beta=beta_,
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
        beta_x, beta_0 = beta_[0], beta_[1 : (self._ns + 1)]
        beta_z = beta_[(self._ns + 1) :] if self._Z is not None else None

        return pd.Series(
            np.r_[
                mu_x,
                sigma2_x,
                a,
                b,
                sigma2_w,
                beta_x,
                beta_0,
                [] if self._Z is None else beta_[(self._ns + 1) :],
            ],
            index=(
                ["mu_x", "sigma2_x"]
                + ["a"] * self._ns
                + ["b"] * self._ns
                + ["sigma2_w"] * self._ns
                + ["beta_x"]
                + ["beta_0"] * self._ns
                + ([] if self._Z is None else ["beta_z"] * self._Z.shape[1])
            ),
        )


def bootstrap_estimator(
    estimator: EM,
    X: ndarray,
    Y: ndarray,
    W: ndarray,
    S: ndarray,
    Z: ndarray | None = None,
    Y_type: Literal["continue", "binary"] = False,
    n_repeat: int = 200,
    seed: int | None | Generator = None,
    pbar: bool = True,
) -> pd.DataFrame:
    assert hasattr(
        estimator, "params_"
    ), "please run regular EM iteration firstly!"

    seed = np.random.default_rng(seed)

    Svals = np.unique(S)
    is_m = pd.isnull(X)
    not_m = ~is_m
    if Y_type == "binary":
        Yvals = np.unique(Y)

    res = []
    estimator._pbar = False  # 将内部的所有tqdm去掉
    init_params = estimator.params_.copy()
    for _ in tqdm(range(n_repeat), desc="Bootstrap: ", disable=not pbar):
        ind_bootstrap = []
        for si in Svals:
            for is_i in [is_m, not_m]:
                if Y_type == "continue":
                    ind = np.nonzero((S == si) & is_i)[0]
                    if len(ind) == 0:
                        continue
                    ind_choice = seed.choice(ind, len(ind), replace=True)
                    ind_bootstrap.append(ind_choice)
                elif Y_type == "binary":
                    for yi in Yvals:
                        ind = np.nonzero((S == si) & (Y == yi) & is_i)[0]
                        if len(ind) == 0:
                            continue
                        ind_choice = seed.choice(ind, len(ind), replace=True)
                        ind_bootstrap.append(ind_choice)
                else:
                    raise NotImplementedError
        ind_bootstrap = np.concatenate(ind_bootstrap)

        Xi, Yi, Wi, Si, Zi = (
            X[ind_bootstrap],
            Y[ind_bootstrap],
            W[ind_bootstrap],
            S[ind_bootstrap],
            None if Z is None else Z[ind_bootstrap],
        )

        estimator.register_data(Xi, Si, Wi, Yi, Zi)
        # 使用EM估计值作为初始值
        estimator.run(init_params=init_params)
        res.append(estimator.params_)

    return pd.concat(res, axis=1).T


class EMBP(BiomarkerPoolBase):

    def __init__(
        self,
        outcome_type: Literal["continue", "binary"],
        max_iter: int = 500,
        max_iter_inner: int = 100,
        delta1: float = 1e-3,
        delta1_inner: float = 1e-4,
        delta2: float = 1e-7,
        delta2_inner: float = 1e-7,
        delta1_var: float = 1e-1,
        delta2_var: float = 1e-3,
        n_importance_sampling: int = 1000,
        lr: float = 1.0,
        variance_estimate: bool = False,
        variance_estimate_method: Literal["sem", "bootstrap"] = "sem",
        n_bootstrap: int = 200,
        pbar: bool = True,
        seed: int | None = 0,
        use_gpu: bool = False,
    ) -> None:
        assert outcome_type in ["continue", "binary"]
        assert variance_estimate_method in ["sem", "bootstrap"]
        if use_gpu:
            try:
                import torch
            except ImportError:
                raise ImportError(
                    "torch is not installed, "
                    "please install torch or BBP[torch]"
                )
        if use_gpu and outcome_type == "continue":
            raise NotImplementedError
        if outcome_type == "binary" and variance_estimate_method == "sem":
            raise NotImplementedError(
                "use bootstrap for outcome_type = binary"
            )

        self.outcome_type_ = outcome_type
        self.max_iter_ = max_iter
        self.max_iter_inner_ = max_iter_inner
        self.delta1_ = delta1
        self.delta1_inner_ = delta1_inner
        self.delta2_ = delta2
        self.delta2_inner_ = delta2_inner
        self.delta1_var_ = delta1_var
        self.delta2_var_ = delta2_var
        self.nIS_ = n_importance_sampling
        self.lr_ = lr
        self.pbar_ = pbar
        self.var_est_ = variance_estimate
        self.var_est_method_ = variance_estimate_method
        self.n_bootstrap_ = n_bootstrap
        self.use_gpu_ = use_gpu
        self.seed_ = np.random.default_rng(seed)

        if use_gpu and seed is not None:
            torch.random.manual_seed(seed)

    @property
    def result_columns(self):
        if not self.var_est_:
            return ["estimate"]
        if self.var_est_method_ == "bootstrap":
            return ["estimate", "CI_1", "CI_2"]
        return ["estimate", "variance(log)", "std(log)", "CI_1", "CI_2"]

    def fit(
        self,
        X: ndarray,
        S: ndarray,
        W: ndarray,
        Y: ndarray,
        Z: ndarray | None = None,
    ) -> None:
        if self.outcome_type_ == "continue":
            self._estimator = ContinueEM(
                max_iter=self.max_iter_,
                max_iter_inner=self.max_iter_inner_,
                delta1=self.delta1_,
                delta1_inner=self.delta1_inner_,
                delta2=self.delta2_,
                delta2_inner=self.delta2_inner_,
                delta1_var=self.delta1_var_,
                delta2_var=self.delta2_var_,
                pbar=self.pbar_,
            )
        elif self.outcome_type_ == "binary":
            if self.use_gpu_:
                from .embp_gpu import BinaryEMTorch

                self._estimator = BinaryEMTorch(
                    max_iter=self.max_iter_,
                    max_iter_inner=self.max_iter_inner_,
                    delta1=self.delta1_,
                    delta1_inner=self.delta1_inner_,
                    delta2=self.delta2_,
                    delta2_inner=self.delta2_inner_,
                    delta1_var=self.delta1_var_,
                    delta2_var=self.delta2_var_,
                    pbar=self.pbar_,
                    device="cuda:0",
                )
            else:
                self._estimator = BinaryEM(
                    max_iter=self.max_iter_,
                    max_iter_inner=self.max_iter_inner_,
                    delta1=self.delta1_,
                    delta1_inner=self.delta1_inner_,
                    delta1_var=self.delta1_var_,
                    delta2=self.delta2_,
                    delta2_inner=self.delta2_inner_,
                    delta2_var=self.delta2_var_,
                    pbar=self.pbar_,
                    lr=self.lr_,
                    n_importance_sampling=self.nIS_,
                    seed=self.seed_,
                )
        self._estimator.register_data(X, S, W, Y, Z)
        self._estimator.run()
        self.params_ = self._estimator.params_.to_frame("estimate")
        self.params_hist_ = self._estimator.params_hist_

        if not self.var_est_:
            return

        if self.var_est_method_ == "bootstrap":
            # 使用boostrap方法
            res_bootstrap = bootstrap_estimator(
                # 使用复制品，而非原始的estimator
                estimator=deepcopy(self._estimator),
                X=X,
                Y=Y,
                W=W,
                S=S,
                Z=Z,
                Y_type=self.outcome_type_,
                n_repeat=self.n_bootstrap_,
                seed=self.seed_,
                pbar=self.pbar_,
            )
            res_ci = np.quantile(
                res_bootstrap.values,
                q=[0.025, 0.975],
                axis=0,
            )
            self.params_["CI_1"] = res_ci[0, :]
            self.params_["CI_2"] = res_ci[1, :]
        else:
            params_var_ = self._estimator.estimate_variance()
            self.params_["variance(log)"] = params_var_
            self.params_["std(log)"] = np.sqrt(params_var_)
            self.params_["CI_1"] = (
                self.params_["estimate"] - 1.96 * self.params_["std(log)"]
            )
            self.params_["CI_2"] = (
                self.params_["estimate"] + 1.96 * self.params_["std(log)"]
            )
            is_sigma2 = self.params_.index.map(
                lambda x: x.startswith("sigma2")
            )
            self.params_.loc[is_sigma2, "CI_1"] = np.exp(
                np.log(self.params_.loc[is_sigma2, "estimate"])
                - 1.96 * self.params_.loc[is_sigma2, "std(log)"]
            )
            self.params_.loc[is_sigma2, "CI_2"] = np.exp(
                np.log(self.params_.loc[is_sigma2, "estimate"])
                + 1.96 * self.params_.loc[is_sigma2, "std(log)"]
            )
