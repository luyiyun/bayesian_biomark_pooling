import logging
from typing import Literal

import pandas as pd
import numpy as np
from scipy.special import expit, log_expit, softmax
from scipy.stats import norm, rv_continuous
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# import pandas as pd
from numpy import ndarray
from .base import BiomarkerPoolBase, check_split_data


logger = logging.getLogger("EMBP")
logger.setLevel(logging.WARNING)
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
formatter = logging.Formatter(
    "[%(name)s][%(levelname)s][%(asctime)s]:%(message)s"
)
ch.setFormatter(formatter)
logger.addHandler(ch)


def ols(x_des, y) -> np.ndarray:
    return np.linalg.inv(x_des.T @ x_des) @ x_des.T @ y


# def calc_xhat(params: dict, WYZ_xUnKnow: list) -> dict:
#     sigma2 = 1 / (
#         params["beta_x"] ** 2 / params["sigma2_y"]
#         + params["b"] ** 2 / params["sigma2_w"]
#         + 1 / params["sigma2_x"]
#     )

#     xhat, xhat2 = [], []
#     for i, item in enumerate(WYZ_xUnKnow):
#         if item is None:
#             xhat.append(np.array([]))
#             xhat2.append(np.array([]))
#             continue

#         Wi, Yi, Zi = item[1:]
#         residual_yi = Yi - params["beta_0"][i]
#         if Zi is not None:
#             residual_yi -= Zi @ params["beta_z"]
#         e_s = (
#             residual_yi * params["beta_x"] / params["sigma2_y"][i]
#             + (Wi - params["a"][i]) * params["b"][i] / params["sigma2_w"][i]
#             + params["mu_x"] / params["sigma2_x"]
#         )
#         xhat.append(e_s * sigma2[i])
#         xhat2.append((e_s * sigma2[i]) ** 2 + sigma2[i])

#     return {"sigma2": sigma2, "xhat": xhat, "xhat2": xhat2}


def logistic(
    Xdes, y, max_iter: int = 100, thre: float = 1e-7, lr: float = 1.0
):
    beta = np.zeros(Xdes.shape[1])

    for i in range(max_iter):
        p = expit(Xdes @ beta)
        grad = np.sum((p - y)[:, None] * Xdes, axis=0)
        diff = np.max(np.abs(grad))
        logger.debug(f"Init Newton-Raphson: iter={i+1} diff={diff:.4f}")
        if diff < thre:
            break
        hessian = (Xdes.T * p * (1 - p)) @ Xdes
        beta -= lr * np.linalg.inv(hessian) @ grad
    else:
        logger.warning(
            f"Init Newton-Raphson (max_iter={max_iter}) doesn't converge"
        )

    return beta


def init_binary(
    Xo: ndarray,
    Yo: ndarray,
    Zo: ndarray | None,
    Wo: ndarray,
    So: ndarray,
    studies: ndarray,
):
    # mu_x和sigma_x
    mu_x = Xo.mean()
    sigma2_x = np.var(Xo, ddof=1)

    # beta
    Xo_des = [np.ones((Xo.shape[0], 1)), Xo[:, None]]
    if Zo is not None:
        Xo_des.append(Zo)
    Xo_des = np.concatenate(Xo_des, axis=1)
    beta = logistic(Xo_des, Yo)

    # a, b, sigma2_w
    a, b, sigma2_w = [], [], []
    for si in studies:
        mask = So == si
        if mask.sum() == 0:
            a.append(0)
            b.append(0)
            sigma2_w.append(1)
            continue
        Xi, Wi = Xo[mask], Wo[mask]
        Xi_des = np.stack([np.ones(Xi.shape[0]), Xi], axis=1)
        abi = ols(Xi_des, Wi)
        sigma2_ws_i = np.mean((Wi - Xi_des @ abi) ** 2)
        a.append(abi[0])
        b.append(abi[1])
        sigma2_w.append(sigma2_ws_i)

    ns = len(studies)
    params = {
        "mu_x": mu_x,
        "sigma2_x": sigma2_x,
        "beta_0": np.full(ns, beta[0]),
        "beta_x": beta[1],
        "a": np.array(a),
        "b": np.array(b),
        "sigma2_w": np.array(sigma2_w),
    }
    if Zo is not None:
        params["beta_z"] = beta[2:]

    return params


def get_lap_apprx(
    Wm: ndarray,
    Ym: ndarray,
    beta_0_long: ndarray,
    beta_x: float,
    beta_z: ndarray | None,
    a_long: ndarray,
    b_long: ndarray,
    sigma2_w_long: ndarray,
    mu_x: float,
    sigma2_x: float,
    Zm: ndarray | None,
    max_iter: int = 100,
    thre: float = 1e-7,
    lr: float = 1.0,
) -> rv_continuous:
    b_long2 = b_long**2
    beta_x2 = beta_x**2
    grad_const = (
        -beta_x * Ym + b_long * (a_long - Wm) / sigma2_w_long - mu_x / sigma2_x
    )
    grad_mul = b_long2 / sigma2_w_long + 1 / sigma2_x

    Z_part = 0.0 if Zm is None else Zm @ beta_z
    delta_part = beta_0_long + Z_part

    Xm = 0  # np.random.randn(Wm.shape[0])
    for i in range(max_iter):
        p = expit(Xm * beta_x + delta_part)
        grad = beta_x * p + grad_mul * Xm + grad_const
        hessian = beta_x2 * p * (1 - p) + grad_mul

        diff = np.max(np.abs(grad))
        logger.debug(f"E step Newton-Raphson: iter={i+1} diff={diff:.4f}")
        if diff < thre:
            break

        Xm -= lr * grad / hessian
    else:
        logger.warning(
            f"E step Newton-Raphson (max_iter={max_iter}) doesn't converge"
        )
    # return multivariate_normal(mean=Xm, cov=1 / hessian)
    return norm(loc=Xm, scale=1 / np.sqrt(hessian))


def newton_raphson_beta(
    init_beta: ndarray,
    Xo_des: ndarray,
    Xm_des: ndarray,
    Yo: ndarray,
    Ym: ndarray,
    wIS: ndarray,
    lr: float = 1.0,
    max_iter: int = 100,
    thre: float = 1e-7,
):
    beta_ = init_beta
    for i in range(max_iter):
        p_o = expit(Xo_des @ beta_)
        p_m = expit(Xm_des @ beta_)
        mul_o = p_o - Yo
        mul_m = (p_m - Ym) * wIS
        grad = np.sum(Xo_des * mul_o[:, None], axis=0) + np.sum(
            Xm_des * mul_m[..., None], axis=(0, 1)
        )

        diff = np.max(np.abs(grad))
        logger.debug(f"M step Newton-Raphson: iter={i+1} diff={diff:.4f}")
        if diff < thre:
            break

        H_o = (Xo_des.T * p_o * (1 - p_o)) @ Xo_des
        H_m = np.sum(
            (Xm_des * (wIS * p_m * (1 - p_m))[..., None]).swapaxes(1, 2)
            @ Xm_des,
            axis=0,
        )
        H = H_o + H_m

        beta_ -= lr * np.linalg.inv(H) @ grad
    else:
        logger.warning(
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

    def prepare(self):
        """
        准备一些在后续步骤中会用到的array
        """
        raise NotImplementedError

    def init(self) -> dict:
        """初始化参数

        Returns:
            dict: 参数组成的dict
        """
        raise NotImplementedError

    def e_step(self, params: dict):
        """Expectation step

        从EM算法的定义上，是计算log joint likelihood的后验期望，也就是Q function。
        但是，在code中一般不是计算这个，而是计算Q function中关于后验期望的部分，
        以便于后面的m step。
        """
        raise NotImplementedError

    def m_step(self, params: dict) -> dict:
        raise NotImplementedError

    def calc_diff(self, params_old: dict, params_new: dict) -> float:
        diff = []
        for k, v in params_new.items():
            if v is None:
                continue
            diffk = params_old[k] - v
            if isinstance(diffk, float):
                diffk = np.array([diffk])
            diff.append(diffk)
        diff = np.concatenate(diff)
        return np.max(np.abs(diff))

    def run(self, pbar: bool = True):
        self.prepare()
        params = self.init()
        with logging_redirect_tqdm(loggers=[logger]):
            for iter_i in tqdm(
                range(1, self._max_iter + 1), desc="EM: ", disable=not pbar
            ):

                self.e_step(params)
                params_new = self.m_step(params)
                diff = self.calc_diff(params, params_new)
                logger.info(
                    f"EM iteration {iter_i}: difference is {diff: .4f}"
                )
                params = params_new  # 更新
                if diff < self._thre:
                    break
            else:
                logger.warning(
                    f"EM iteration (max_iter={self._max_iter}) "
                    "doesn't converge"
                )

        self.params_ = params


class ContinueEM(EM):

    def prepare(self):
        # 准备后悔步骤中会用到的array，预先计算，节省效率
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
        # dats = check_split_data(X, S, W, Y, Z)
        # ind_o = dats["ind_o"]
        # ind_s = dats["ind_s"]
        # ind_s_inv = dats["ind_s_inv"]
        # n_ms, n_os = dats["n_ms"], dats["n_os"]
        # n_s = n_ms + n_os

        self._Xhat = np.copy(self._X)
        self._Xhat2 = self._Xhat**2

    def init(self) -> dict:
        # 初始化权重
        mu_x = self._Xo.mean()
        sigma2_x = np.var(self._Xo, ddof=1)

        Xo_des = [np.ones((self._Xo.shape[0], 1)), self._Xo[:, None]]
        if self._Z is not None:
            Xo_des.append(self._Zo)
        Xo_des = np.concatenate(Xo_des, axis=1)
        beta = ols(Xo_des, self._Yo)
        sigma2_ys = np.mean((self._Yo - Xo_des @ beta) ** 2)

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

        return {
            "mu_x": mu_x,
            "sigma2_x": sigma2_x,
            "beta_0": np.full(self._ns, beta[0]),
            "beta_x": beta[1],
            "sigma2_y": np.full(self._ns, sigma2_ys),
            "a": np.array(a),
            "b": np.array(b),
            "sigma2_w": np.array(sigma2_w),
            "beta_z": None if self._Z is None else beta[2:],
        }

    def e_step(self, params: dict):
        self._sigma2 = 1 / (
            params["beta_x"] ** 2 / params["sigma2_y"]
            + params["b"] ** 2 / params["sigma2_w"]
            + 1 / params["sigma2_x"]
        )  # 最后在迭代计算sigma2_y的时候还会用到
        z_m_part = 0.0 if self._Z is None else self._Zm @ params["beta_z"]
        beta_0_m_long = params["beta_0"][self._ind_m_inv]
        sigma2_y_m_long = params["sigma2_y"][self._ind_m_inv]
        a_m_long = params["a"][self._ind_m_inv]
        b_m_long = params["b"][self._ind_m_inv]
        sigma2_w_m_long = params["sigma2_w"][self._ind_m_inv]
        sigma2_m_long = self._sigma2[self._ind_m_inv]

        xhat_m = (
            (self._Ym - beta_0_m_long - z_m_part)
            * params["beta_x"]
            / sigma2_y_m_long
            + (self._Wm - a_m_long) * b_m_long / sigma2_w_m_long
            + params["mu_x"] / params["sigma2_x"]
        ) * sigma2_m_long

        self._Xhat[self._is_m] = xhat_m
        self._Xhat2[self._is_m] = xhat_m**2 + sigma2_m_long

    def m_step(self, params: dict) -> dict:
        vbar = self._Xhat2.mean()
        wxbar_s = np.array(
            [np.mean(self._W[ind] * self._Xhat[ind]) for ind in self._ind_S]
        )
        vbar_s = np.array([np.mean(self._Xhat2[ind]) for ind in self._ind_S])
        xbar_s = np.array([np.mean(self._Xhat[ind]) for ind in self._ind_S])

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
        beta_x, beta_0, sigma2_y, beta_z = self.iter_calc_params(
            params["beta_x"],
            params["beta_0"],
            params["sigma2_y"],
            params["beta_z"],
        )

        return {
            "mu_x": mu_x,
            "sigma2_x": sigma2_x,
            "a": a,
            "b": b,
            "sigma2_w": sigma2_w,
            "beta_x": beta_x,
            "beta_0": beta_0,
            "sigma2_y": sigma2_y,
            "beta_z": beta_z,
        }

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

            logger.info(f"Inner iteration {iter_i}: difference is {diff: .4f}")

            beta_x = beta_x_new
            beta_0 = beta_0_new
            sigma2_y = sigma2_y_new
            if self._Z is not None:
                beta_z = beta_z_new

            if diff < self._thre_inner:
                logger.info(f"Inner iteration stop, stop iter: {iter_i}")
                break

        return beta_x, beta_0, sigma2_y, beta_z


# TODO: 控制一下内部的Newton-Raphson的max_iter和thre
class EMBP(BiomarkerPoolBase):

    def __init__(
        self,
        outcome_type: Literal["continue", "binary"],
        thre: float = 1e-5,
        max_iter: int = 100,
        thre_inner: float = 1e-7,
        max_iter_inner: int = 100,
        nsample_IS: int = 10000,
        lr: float = 1.0,
        pbar: bool = True,
    ) -> None:
        assert outcome_type in ["continue", "binary"]

        self.outcome_type_ = outcome_type
        self.thre_ = thre
        self.max_iter_ = max_iter
        self.thre_inner_ = thre_inner
        self.max_iter_inner_ = max_iter_inner
        self.nsample_IS_ = nsample_IS
        self.lr_ = lr
        self.pbar_ = pbar

    def fit(
        self,
        X: ndarray,
        S: ndarray,
        W: ndarray,
        Y: ndarray,
        Z: ndarray | None = None,
    ) -> None:
        if self.outcome_type_ == "continue":
            estimator = ContinueEM(
                X,
                S,
                W,
                Y,
                Z,
                self.max_iter_,
                self.thre_,
                self.max_iter_inner_,
                self.thre_inner_,
            )
            estimator.run(self.pbar_)
        elif self.outcome_type_ == "binary":
            self._fit_binary(X, S, W, Y, Z)
        else:
            raise NotImplementedError

        self.params_ = estimator.params_

    # def _fit_continue(
    #     self,
    #     X: ndarray,
    #     S: ndarray,
    #     W: ndarray,
    #     Y: ndarray,
    #     Z: ndarray | None = None,
    # ) -> None:
    #     n = X.shape[0]
    #     dats = check_split_data(X, S, W, Y, Z)
    #     ind_o = dats["ind_o"]
    #     ind_s = dats["ind_s"]
    #     ind_s_inv = dats["ind_s_inv"]
    #     n_ms, n_os = dats["n_ms"], dats["n_os"]
    #     n_s = n_ms + n_os

    #     Xhat = np.copy(X)
    #     Xhat2 = Xhat**2

    #     # 1. 使用OLS得到初始值
    #     params = init_continue(
    #         X[ind_o],
    #         Y[ind_o],
    #         None if Z is None else Z[ind_o, :],
    #         dats["XWYZ_xKnow"],
    #     )

    #     for iter_i in range(self.max_iter_):

    #         # 2. E step, 计算hat x
    #         E_res = calc_xhat(params, dats["WYZ_xUnKnow"])
    #         sigma2 = E_res["sigma2"]
    #         for xhat_m, xhat2_m, item in zip(
    #             E_res["xhat"], E_res["xhat2"], dats["WYZ_xUnKnow"]
    #         ):
    #             if item is None:
    #                 continue
    #             ind_m_s = item[0]
    #             Xhat[ind_m_s] = xhat_m
    #             Xhat2[ind_m_s] = xhat2_m

    #         # 3. M step，更新参数值
    #         mu_x = np.mean(Xhat)
    #         sigma2_x = np.mean((Xhat - mu_x) ** 2) + np.sum(sigma2 * n_s / n)
    #         a, b, sigma2_w = [], [], []
    #         for i, ind_si in enumerate(ind_s):
    #             wsi, xsi, x2si = W[ind_si], Xhat[ind_si], Xhat2[ind_si]
    #             wsi_ = wsi.mean()
    #             xsi_ = xsi.mean()
    #             x2si_ = x2si.mean()

    #             bsi = ((wsi * xsi).mean() - wsi_ * xsi_) / (x2si_ - xsi_**2)
    #             resid = wsi - bsi * xsi
    #             asi = np.mean(resid)
    #             sigma2_wsi = np.mean((resid - asi) ** 2)
    #             b.append(bsi)
    #             a.append(asi)
    #             sigma2_w.append(sigma2_wsi)
    #         a, b, sigma2_w = np.array(a), np.array(b), np.array(sigma2_w)
    #         sigma2_w += b**2 * sigma2 * n_ms / n_s

    #         # 迭代法求解剩余的参数
    #         beta_x, beta_0, sigma2_y, beta_z = iter_calc_params(
    #             Y=Y,
    #             Z=Z,
    #             Xhat=Xhat,
    #             Xhat2=Xhat2,
    #             sigma2=sigma2,
    #             ind_s=ind_s,
    #             ind_s_inv=ind_s_inv,
    #             n_ms=n_ms,
    #             n_s=n_s,
    #             beta_x=params["beta_x"],
    #             beta_0=params["beta_0"],
    #             beta_z=params.get("beta_z", 0),
    #             sigma2_y=params["sigma2_y"],
    #             thre=self.thre_inner_,
    #             max_iter=self.max_iter_inner_,
    #         )

    #         # 4. 检查是否收敛
    #         diff = np.r_[
    #             mu_x - params["mu_x"],
    #             sigma2_x - params["sigma2_x"],
    #             beta_x - params["beta_x"],
    #             beta_0 - params["beta_0"],
    #             sigma2_y - params["sigma2_y"],
    #             a - params["a"],
    #             b - params["b"],
    #             sigma2_w - params["sigma2_w"],
    #         ]
    #         if Z is not None:
    #             diff = np.r_[diff, beta_z - params["beta_z"]]
    #         diff = np.max(np.abs(diff))
    #         logger.info(
    #             f"Outer iteration {iter_i+1}: difference is {diff: .4f}"
    #         )

    #         params["mu_x"] = mu_x
    #         params["sigma2_x"] = sigma2_x
    #         params["beta_x"] = beta_x
    #         params["beta_0"] = beta_0
    #         params["sigma2_y"] = sigma2_y
    #         params["a"] = a
    #         params["b"] = b
    #         params["sigma2_w"] = sigma2_w
    #         if Z is not None:
    #             params["beta_z"] = beta_z

    #         if diff < self.thre_:
    #             logger.info(f"Outer iteration stop: difference is {diff: .4f}")
    #             break

    #     self.params_ = params

    def _fit_binary(
        self,
        X: ndarray,
        S: ndarray,
        W: ndarray,
        Y: ndarray,
        Z: ndarray | None = None,
    ) -> None:

        studies, s_ind_v = np.unique(
            S, return_index=False, return_inverse=True
        )
        len_s = len(studies)

        is_m = pd.isnull(X)
        is_o = ~is_m
        n_m = is_m.sum()
        n_o = is_o.sum()

        ind_S = [np.nonzero(S == s)[0] for s in studies]
        is_Sm = [np.nonzero(S[is_m] == s)[0] for s in studies]
        is_So = [np.nonzero(S[is_o] == s)[0] for s in studies]
        # ns = np.array([len(ind_s) for ind_s in ind_S])
        Co = np.zeros((n_o, len_s))
        Cm = np.zeros((self.nsample_IS_, n_m, len_s))
        for i in range(len_s):
            Co[is_So[i], i] = 1
            Cm[:, is_Sm[i], i] = 1

        Wm, Ym = W[is_m], Y[is_m]
        Xo, Yo = X[is_o], Y[is_o]
        xhat = np.copy(X)
        vhat = xhat**2

        Z_m = None if Z is None else Z[is_m, :]
        Z_o = None if Z is None else Z[is_o, :]
        Xo_des = [Xo[:, None], Co]  # 360 * 5
        if Z is not None:
            Xo_des.append(Z_o)
        Xo_des = np.concatenate(Xo_des, axis=-1)

        # 1. 通过Newton-Raphson和OLS计算初始值
        params = init_binary(
            Xo,
            Yo,
            None if Z is None else Z[is_o],
            W[is_o],
            S[is_o],
            studies,
        )

        for iter_i in range(self.max_iter_):
            Z_part = 0.0 if Z is None else Z @ params["beta_z"]
            Z_part_m = 0.0 if Z is None else Z_part[is_m, :]
            # Z_part_o = 0.0 if Z is None else Z_part[is_o, :]

            # 2. E step, 首先通过Newton-Raphson得到proposed distribution，然后
            # 进行IS采样
            beta_0_long = params["beta_0"][s_ind_v]
            a_long = params["a"][s_ind_v]
            b_long = params["b"][s_ind_v]
            sigma2_w_long = params["sigma2_w"][s_ind_v]

            beta_0_long_m = beta_0_long[is_m]
            a_long_m = a_long[is_m]
            b_long_m = b_long[is_m]
            sigma2_w_long_m = sigma2_w_long[is_m]
            # beta_0_long_o = beta_0_long[is_o]
            # a_long_o = a_long[is_o]
            # b_long_o = b_long[is_o]
            # sigma2_w_long_o = sigma2_w_long[is_o]

            P_lap = get_lap_apprx(
                Wm,
                Ym,
                beta_0_long_m,
                params["beta_x"],
                params.get("beta_z", None),
                a_long_m,
                b_long_m,
                sigma2_w_long_m,
                params["mu_x"],
                params["sigma2_x"],
                None if Z is None else Z[is_m, :],
            )
            xIS = P_lap.rvs(size=(self.nsample_IS_, n_m))  # N x n_m
            pIS = log_expit(
                (2 * Ym - 1)
                * (beta_0_long_m + params["beta_x"] * xIS + Z_part_m)
            )
            pIS -= 0.5 * (
                (Wm - a_long_m - b_long_m * xIS) ** 2 / sigma2_w_long_m
                + (xIS - params["beta_x"]) ** 2 / params["sigma2_x"]
            )
            pIS = pIS - P_lap.logpdf(xIS)
            wIS = softmax(pIS, axis=0)
            Seff = 1 / np.sum(wIS**2, axis=0)
            logger.debug(
                "Importance effective size "
                + f"is {Seff.mean():.2f}±{Seff.std():.2f}"
            )

            xhat[is_m] = np.sum(xIS * wIS, axis=0)
            vhat[is_m] = np.sum(xIS**2 * wIS, axis=0)
            vbar = vhat.mean()

            xbars = np.array([xhat[sind].mean() for sind in ind_S])
            vbars = np.array([vhat[sind].mean() for sind in ind_S])
            wbars = np.array([W[sind].mean() for sind in ind_S])
            wwbars = np.array([(W[sind] ** 2).mean() for sind in ind_S])
            wxbars = np.array(
                [np.mean(W[sind] * xhat[sind]) for sind in ind_S]
            )

            # 3. M step
            mu_x = xhat.mean()
            sigma2_x = vbar - mu_x**2
            b = (wxbars - wbars * xbars) / (vbars - xbars**2)
            a = wbars - b * xbars
            sigma2_w = (
                wwbars
                + vbars * b**2
                + a**2
                - 2 * (a * wbars + b * wxbars - a * b * xbars)
            )

            beta_ = np.r_[params["beta_x"], params["beta_0"]]  # 1+S+p
            if Z is not None:
                beta_ = np.r_[beta_, params["beta_z"]]
            Xm_des = [xIS[:, :, None], Cm]  # 1000 * 360 * 5
            if Z is not None:
                Xm_des.append(Z_m[None, ...])
            Xm_des = np.concatenate(Xm_des, axis=-1)

            beta_ = newton_raphson_beta(
                beta_, Xo_des, Xm_des, Yo, Ym, wIS, self.lr_
            )
            beta_x, beta_0 = beta_[0], beta_[1 : (len_s + 1)]
            if Z is not None:
                beta_z = beta_[(len_s + 1) :]

            # 4. 检查是否收敛
            diff = np.r_[
                mu_x - params["mu_x"],
                sigma2_x - params["sigma2_x"],
                beta_x - params["beta_x"],
                beta_0 - params["beta_0"],
                a - params["a"],
                b - params["b"],
                sigma2_w - params["sigma2_w"],
            ]
            if Z is not None:
                diff = np.r_[diff, beta_z - params["beta_z"]]
            diff = np.max(np.abs(diff))
            logger.debug(f"EM iteration {iter_i}: difference is {diff: .4f}")

            params["mu_x"] = mu_x
            params["sigma2_x"] = sigma2_x
            params["beta_x"] = beta_x
            params["beta_0"] = beta_0
            params["a"] = a
            params["b"] = b
            params["sigma2_w"] = sigma2_w
            if Z is not None:
                params["beta_z"] = beta_z

            if diff < self.thre_:
                break
        else:
            logger.warning(
                f"EM iteration (max_iter={self.max_iter_}) doesn't converge"
            )

        self.params_ = params
