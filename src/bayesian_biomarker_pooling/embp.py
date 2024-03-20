import logging
from typing import Literal

import pandas as pd
import numpy as np
from scipy.special import expit, log_expit, softmax
from scipy.stats import norm
from scipy.linalg import block_diag
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# import pandas as pd
from numpy import ndarray

from .base import BiomarkerPoolBase
from .logger import logger_embp


def ols(x_des, y) -> np.ndarray:
    return np.linalg.inv(x_des.T @ x_des) @ x_des.T @ y


def logistic(
    Xdes, y, max_iter: int = 100, thre: float = 1e-7, lr: float = 1.0
):
    beta = np.zeros(Xdes.shape[1])

    for i in range(max_iter):
        p = expit(Xdes @ beta)
        grad = Xdes.T @ (p - y)
        hessian = (Xdes.T * p * (1 - p)) @ Xdes
        delta = lr * np.linalg.inv(hessian) @ grad
        beta -= delta

        diff = np.max(np.abs(delta))
        logger_embp.info(f"Init Newton-Raphson: iter={i+1} diff={diff:.4f}")
        if diff < thre:
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
    thre: float = 1e-7,
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

        diff = np.max(np.abs(beta_delta))
        logger_embp.info(f"M step Newton-Raphson: iter={i+1} diff={diff:.4f}")
        if diff < thre:
            break
    else:
        logger_embp.warning(
            f"M step Newton-Raphson (max_iter={max_iter}) doesn't converge"
        )

    return beta_


class EM:

    def __init__(
        self,
        X: ndarray,
        S: ndarray,
        W: ndarray,
        Y: ndarray,
        Z: ndarray | None = None,
        max_iter: int = 100,
        thre: float = 1e-5,
        max_iter_inner: int = 100,
        thre_inner: float = 1e-7,
        pbar: bool = True,
        thre_var_est: float = 1e-3,
    ) -> None:
        self._X = X
        self._S = S
        self._W = W
        self._Y = Y
        self._Z = Z

        self._max_iter = max_iter
        self._thre = thre
        self._max_iter_inner = max_iter_inner
        self._thre_inner = thre_inner
        self._pbar = pbar
        self._thre_var_est = thre_var_est

    def prepare(self):
        # 准备后续步骤中会用到的array，预先计算，节省效率
        self._n = self._Y.shape[0]
        self._studies, self._ind_inv = np.unique(self._S, return_inverse=True)
        self._is_m = pd.isnull(self._X)
        self._is_o = ~self._is_m
        self._ns = len(self._studies)
        self._n_o = self._is_o.sum()
        self._n_m = self._is_m.sum()

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

    def e_step(self, params: pd.Series):
        """Expectation step

        从EM算法的定义上，是计算log joint likelihood的后验期望，也就是Q function。
        但是，在code中一般不是计算这个，而是计算Q function中关于后验期望的部分，
        以便于后面的m step。
        """
        raise NotImplementedError

    def m_step(self, params: pd.Series) -> pd.Series:
        raise NotImplementedError

    def run(self):
        self.prepare()

        params = self.init()
        self.params_hist_ = [params]
        with logging_redirect_tqdm(loggers=[logger_embp]):
            for iter_i in tqdm(
                range(1, self._max_iter + 1),
                desc="EM: ",
                disable=not self._pbar,
            ):

                self.e_step(params)
                params_new = self.m_step(params)
                diff = np.max(np.abs(params - params_new))
                logger_embp.info(
                    f"EM iteration {iter_i}: difference is {diff: .4f}"
                )
                params = params_new  # 更新
                self.params_hist_.append(params)
                if diff < self._thre:
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

        finish_row_ind = []
        R = []
        with logging_redirect_tqdm(loggers=[logger_embp]):
            for t in tqdm(
                range(self.params_hist_.shape[0]),
                desc="Estimate Variance: ",
                disable=not self._pbar,
            ):
                params_i = self.params_hist_.iloc[t, :]
                Rt = []
                for j in range(n_params):

                    # 如果某一行已经收敛，则不行再去进行计算了
                    if t > 0 and j in finish_row_ind:
                        Rt.append(R[-1][j, :])
                        continue

                    inpt = self.params_.copy()
                    x = inpt.iloc[j] = params_i.iloc[j]

                    self.e_step(inpt)
                    oupt = self.m_step(inpt)

                    # 修改sigma2为log尺度
                    oupt.iloc[ind_sigma2] = np.log(oupt.iloc[ind_sigma2])
                    if j in ind_sigma2:
                        x = np.log(x)

                    # 计算差值比来作为导数的估计
                    Rt.append(
                        (oupt.values - params_w_log.values)
                        / (x - params_w_log.iloc[j])
                    )
                Rt = np.stack(Rt, axis=0)

                # 看一下有哪些行完成了收敛
                if t > 0:
                    finish_row_ind = np.nonzero(
                        np.max(np.abs(Rt - R[-1]), axis=1) < self._thre_var_est
                    )[0]

                    logger_embp.info("finished_row:" + str(finish_row_ind))

                R.append(Rt)
                if len(finish_row_ind) == n_params:
                    break
            else:
                logger_embp.warn("estimate variance does not converge.")

        self._R = np.stack(R, axis=0)
        DM = self._R[-1]

        v_joint = self.v_joint(self.params_)
        self.params_cov_ = v_joint + v_joint @ DM @ np.linalg.inv(
            np.diag(np.ones(n_params)) - DM
        )
        return np.diag(self.params_cov_)


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
                    + ["beta_z"] * (len(beta) - 2)
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
        # beta_x, beta_0, sigma2_y, beta_z = self.iter_calc_params(
        #     params["beta_x"],
        #     params["beta_0"].values,
        #     params["sigma2_y"].values,
        #     params["beta_z"].values if self._Z is not None else None,
        # )
        beta_x = params["beta_x"]
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

            diff = np.max(
                np.abs(
                    np.r_[
                        beta_x_new - beta_x,
                        beta_0_new - beta_0,
                        sigma2_y_new - sigma2_y,
                        [] if self._Z is None else beta_z_new - beta_z,
                    ]
                )
            )

            logger_embp.info(
                f"Inner iteration {i+1}: difference is {diff: .4f}"
            )
            beta_x = beta_x_new
            beta_0 = beta_0_new
            sigma2_y = sigma2_y_new
            if self._Z is not None:
                beta_z = beta_z_new

            if diff < self._thre_inner:
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
                beta_x,
                beta_0,
                [] if self._Z is None else beta_z,
                sigma2_y,
            ],
            index=(
                ["mu_x", "sigma2_x"]
                + ["a"] * self._ns
                + ["b"] * self._ns
                + ["sigma2_w"] * self._ns
                + ["beta_x"]
                + ["beta_0"] * self._ns
                + ([] if self._Z is None else ["beta_z"] * self._Z.shape[1])
                + ["sigma2_y"] * self._ns
            ),
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
            B = (self._n * xzbar)[None, :]
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
                - 2 * beta_0 * xbar_s
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
        X: ndarray,
        S: ndarray,
        W: ndarray,
        Y: ndarray,
        Z: ndarray | None = None,
        max_iter: int = 100,
        thre: float = 0.00001,
        max_iter_inner: int = 100,
        thre_inner: float = 1e-7,
        thre_var_est: float = 1e-3,
        pbar: bool = True,
        lr: float = 1.0,
        nsample_IS: int = 1000,
        ema: float = 0.5,  # 指数滑动平均的权重，1表示只用当前值
    ) -> None:
        assert ema >= 0 and ema <= 1
        super().__init__(
            X=X,
            S=S,
            W=W,
            Y=Y,
            Z=Z,
            max_iter=max_iter,
            thre=thre,
            max_iter_inner=max_iter_inner,
            thre_inner=thre_inner,
            thre_var_est=thre_var_est,
            pbar=pbar,
        )
        self._lr = lr
        self._nIS = nsample_IS
        self._ema = ema

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
            max_iter=self._max_iter_inner,
            thre=self._max_iter_inner,
            lr=self._lr,
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

        Xm = 0  # np.random.randn(Wm.shape[0])
        for i in range(1, self._max_iter_inner + 1):
            p = expit(Xm * beta_x + delta_part)
            grad = beta_x * p + grad_mul * Xm + grad_const
            hessian = beta_x_2 * p * (1 - p) + grad_mul

            xdelta = self._lr * grad / hessian
            Xm -= xdelta

            diff = np.max(np.abs(xdelta))
            logger_embp.info(
                f"E step Newton-Raphson: iter={i} diff={diff:.4f}"
            )
            if diff < self._thre_inner:
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
        self._XIS = norm_lap.rvs(size=(self._nIS, self._n_m))  # N x n_m

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
            thre=self._thre_inner,
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

    def run(self):  # 要使用滑动平均技术
        self.prepare()

        params_ema = params = self.init()
        self.params_hist_ori_ = [params]
        self.params_hist_ = [params]
        with logging_redirect_tqdm(loggers=[logger_embp]):
            for iter_i in tqdm(
                range(1, self._max_iter + 1),
                desc="EM: ",
                disable=not self._pbar,
            ):

                self.e_step(params)
                params = self.m_step(params)
                self.params_hist_ori_.append(params)

                params_ema = self._ema * params + (1 - self._ema) * params_ema
                self.params_hist_.append(params_ema)

                diff = np.max(np.abs(params_ema - self.params_hist_[-2]))
                logger_embp.info(
                    f"EM iteration {iter_i}: difference is {diff: .4f}"
                )
                if diff < self._thre:
                    self.iter_convergence_ = iter_i
                    break
            else:
                logger_embp.warning(
                    f"EM iteration (max_iter={self._max_iter}) "
                    "doesn't converge"
                )

        self.params_ = params_ema
        self.params_hist_ori_ = pd.concat(self.params_hist_ori_, axis=1).T
        self.params_hist_ = pd.concat(self.params_hist_, axis=1).T


class EMBP(BiomarkerPoolBase):

    def __init__(
        self,
        outcome_type: Literal["continue", "binary"],
        thre: float = 1e-5,
        max_iter: int = 100,
        thre_inner: float = 1e-7,
        max_iter_inner: int = 100,
        nsample_IS: int = 1000,
        lr: float = 1.0,
        variance_estimate: bool = False,
        variance_esitmate_method: Literal["sem", "boostrap"] = "sem",
        thre_var_est: float = 1e-4,
        boostrap_samples: int = 200,
        pbar: bool = True,
        seed: int | None = 0,
        ema: float = 0.1,
        use_gpu: bool = False,
    ) -> None:
        assert outcome_type in ["continue", "binary"]
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

        self.outcome_type_ = outcome_type
        self.thre_ = thre
        self.max_iter_ = max_iter
        self.thre_inner_ = thre_inner
        self.max_iter_inner_ = max_iter_inner
        self.nsample_IS_ = nsample_IS
        self.lr_ = lr
        self.pbar_ = pbar
        self.var_est_ = variance_estimate
        self.var_est_method_ = variance_esitmate_method
        self.thre_var_est_ = thre_var_est
        self.boostrap_samples_ = boostrap_samples
        self.ema_ = ema
        self.use_gpu_ = use_gpu

        if use_gpu:
            torch.random.manual_seed(seed)
        else:
            self._rng = np.random.default_rng(seed)

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
                X,
                S,
                W,
                Y,
                Z,
                self.max_iter_,
                self.thre_,
                self.max_iter_inner_,
                self.thre_inner_,
                pbar=self.pbar_,
                thre_var_est=self.thre_var_est_,
            )
        elif self.outcome_type_ == "binary":
            if self.use_gpu_:
                from .embp_gpu import BinaryEMTorch

                self._estimator = BinaryEMTorch(
                    X,
                    S,
                    W,
                    Y,
                    Z,
                    self.max_iter_,
                    self.thre_,
                    self.max_iter_inner_,
                    self.thre_inner_,
                    pbar=self.pbar_,
                    lr=self.lr_,
                    nsample_IS=self.nsample_IS_,
                    thre_var_est=self.thre_var_est_,
                    ema=self.ema_,
                    device="cuda:0",
                )
            else:
                self._estimator = BinaryEM(
                    X,
                    S,
                    W,
                    Y,
                    Z,
                    self.max_iter_,
                    self.thre_,
                    self.max_iter_inner_,
                    self.thre_inner_,
                    pbar=self.pbar_,
                    lr=self.lr_,
                    nsample_IS=self.nsample_IS_,
                    thre_var_est=self.thre_var_est_,
                    ema=self.ema_,
                )
        else:
            raise NotImplementedError
        self._estimator.run()
        self.params_ = self._estimator.params_.to_frame("estimate")

        if not self.var_est_:
            return

        if self.var_est_method_ == "sem":
            params_var_ = self._estimator.estimate_variance()
        elif self.var_est_method_ == "boostrap":
            raise NotImplementedError
            # params_bootstrap = []
            # with logging_redirect_tqdm(loggers=[logger]):
            #     for _ in tqdm(
            #         range(self.boostrap_samples_), disable=not self.pbar_
            #     ):
            #         ind = self._rng.choice(
            #             Y.shape[0], size=Y.shape[0], replace=True
            #         )
            #         X_, S_, W_, Y_, Z_ = (
            #             X[ind],
            #             S[ind],
            #             W[ind],
            #             Y[ind],
            #             None if Z is None else Z[ind],
            #         )
            #         if self.outcome_type_ == "continue":
            #             self._estimator = ContinueEM(
            #                 X_,
            #                 S_,
            #                 W_,
            #                 Y_,
            #                 Z_,
            #                 self.max_iter_,
            #                 self.thre_,
            #                 self.max_iter_inner_,
            #                 self.thre_inner_,
            #                 pbar=False,
            #                 thre_var_est=self.thre_var_est_,
            #             )
            #         elif self.outcome_type_ == "binary":
            #             self._estimator = BinaryEM(
            #                 X_,
            #                 S_,
            #                 W_,
            #                 Y_,
            #                 Z_,
            #                 self.max_iter_,
            #                 self.thre_,
            #                 self.max_iter_inner_,
            #                 self.thre_inner_,
            #                 pbar=False,
            #                 lr=self.lr_,
            #                 nsample_IS=self.nsample_IS_,
            #                 thre_var_est=self.thre_var_est_,
            #                 ema=self.ema_,
            #             )
            #         else:
            #             raise NotImplementedError
            #         self._estimator.run()
            #         params_bootstrap.append(self._estimator.params_.values)
            # params_bootstrap = np.stack(params_bootstrap)
            # params_var_ = np.var(params_bootstrap, axis=0, ddof=1)
        self.params_["variance(log)"] = params_var_
        self.params_["std(log)"] = np.sqrt(params_var_)
        self.params_["CI_1"] = (
            self.params_["estimate"] - 1.96 * self.params_["std(log)"]
        )
        self.params_["CI_2"] = (
            self.params_["estimate"] + 1.96 * self.params_["std(log)"]
        )
        is_sigma2 = self.params_.index.map(lambda x: x.startswith("sigma2"))
        self.params_.loc[is_sigma2, "CI_1"] = np.exp(
            np.log(self.params_.loc[is_sigma2, "estimate"])
            - 1.96 * self.params_.loc[is_sigma2, "std(log)"]
        )
        self.params_.loc[is_sigma2, "CI_2"] = np.exp(
            np.log(self.params_.loc[is_sigma2, "estimate"])
            + 1.96 * self.params_.loc[is_sigma2, "std(log)"]
        )
