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

from .base import BiomarkerPoolBase


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


def logistic(
    Xdes, y, max_iter: int = 100, thre: float = 1e-7, lr: float = 1.0
):
    beta = np.zeros(Xdes.shape[1])

    for i in range(max_iter):
        p = expit(Xdes @ beta)
        grad = np.sum((p - y)[:, None] * Xdes, axis=0)
        diff = np.max(np.abs(grad))
        logger.info(f"Init Newton-Raphson: iter={i+1} diff={diff:.4f}")
        if diff < thre:
            break
        hessian = (Xdes.T * p * (1 - p)) @ Xdes
        beta -= lr * np.linalg.inv(hessian) @ grad
    else:
        logger.warning(
            f"Init Newton-Raphson (max_iter={max_iter}) doesn't converge"
        )

    return beta


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
        pbar: bool = True,
        variance_estimate: bool = True,
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
        self._var_est = variance_estimate

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
            [mu_x, sigma2_x] + a + b + [sigma2_w],
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

    def v_joint(self, params: pd.Series) -> ndarray:
        raise NotImplementedError

    def calc_diff(self, params_old: pd.Series, params_new: pd.Series) -> float:
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

    def estimate_variance(self):
        v_joint = self.v_joint(self.params_)
        n_params = sum(
            [
                1 if isinstance(v, float) else len(v)
                for v in self.params_.values()
            ]
        )
        params_star = self.concat_params(self.params_)

        with logging_redirect_tqdm(loggers=[logger]):
            for i in tqdm(
                range(self.iter_convergence_),
                desc="Estimate Variance: ",
                disable=not self._pbar,
            ):
                params_i = {k: v[i] for k, v in self.params_hist_.items()}
                for k, v in self.params_.items():
                    if v is None:
                        continue
                    if isinstance(v, float):
                        inpt = deepcopy(self.params_)
                        inpt[k] = self.params_hist_[k][i]

                        self.e_step(inpt)
                        oupt = self.m_step(inpt)

                        x_diff = (
                            np.log(inpt[k]) - np.log(self.params_[k])
                            if k.startswith("sigma2")
                            else inpt[k] - self.params_[k]
                        )
                        rt_col = (
                            self.concat_params(oupt, log_sigma2=True)
                            - params_star
                        ) / x_diff
                    else:
                        # TODO:
                        pass
                        vs = v
            #     self.e_step(params)
            #     params_new = self.m_step(params)
            #     diff = self.calc_diff(params, params_new)
            #     logger.info(
            #         f"EM iteration {iter_i}: difference is {diff: .4f}"
            #     )
            #     params = params_new  # 更新
            #     if diff < self._thre:
            #         break
            # else:
            #     logger.warning(
            #         f"EM iteration (max_iter={self._max_iter}) "
            #         "doesn't converge"
            #     )
        import ipdb

        ipdb.set_trace()

    def run(self):
        self.prepare()

        params = self.init()
        self.params_hist_ = {
            k: [v] for k, v in params.items() if v is not None
        }
        with logging_redirect_tqdm(loggers=[logger]):
            for iter_i in tqdm(
                range(1, self._max_iter + 1),
                desc="EM: ",
                disable=not self._pbar,
            ):

                self.e_step(params)
                params_new = self.m_step(params)
                diff = self.calc_diff(params, params_new)
                logger.info(
                    f"EM iteration {iter_i}: difference is {diff: .4f}"
                )
                params = params_new  # 更新
                for k in self.params_hist_.keys():
                    self.params_hist_[k].append(params[k])
                if diff < self._thre:
                    self.iter_convergence_ = iter_i
                    break
            else:
                logger.warning(
                    f"EM iteration (max_iter={self._max_iter}) "
                    "doesn't converge"
                )

        self.params_ = params
        self.params_hist_ = {
            k: np.stack(arrs, axis=0) for k, arrs in self.params_hist_.items()
        }

        if self._var_est:
            self.estimate_variance()


class ContinueEM(EM):

    def prepare(self):
        super().prepare()

        if not self._var_est:
            return

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

        self._ybar_s = np.array([self._Y[ind].mean() for ind in self._ind_S])
        self._yybar_s = np.array(
            [(self._Y[ind] ** 2).mean() for ind in self._ind_S]
        )

    def init(self) -> dict:
        params = super().init()

        Xo_des = [np.ones((self._Xo.shape[0], 1)), self._Xo[:, None]]
        if self._Z is not None:
            Xo_des.append(self._Zo)
        Xo_des = np.concatenate(Xo_des, axis=1)
        beta = ols(Xo_des, self._Yo)
        sigma2_ys = np.mean((self._Yo - Xo_des @ beta) ** 2)

        params["beta_0"] = np.full(self._ns, beta[0])
        params["beta_x"] = beta[1]
        params["sigma2_y"] = np.full(self._ns, sigma2_ys)
        params["beta_z"] = None if self._Z is None else beta[2:]

        return params

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

    def concat_params(self, params: dict, log_sigma2: bool = True) -> ndarray:
        arr = []
        for key in [
            "mu_x",
            "sigma2_x",
            "a",
            "b",
            "sigma2_w",
            "beta_x",
            "beta_0",
            "beta_z",
            "sigma2_y",
        ]:
            value = params["mu_x"]
            if key.startswith("sigma2") and log_sigma2:
                value = np.log(value)
            if isinstance(value, np.ndarray):
                arr.extend(value.tolist())
            else:
                arr.append(value)
        return np.array(arr)

    def v_joint(self, params: dict) -> ndarray:

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

        x12 = (xbar - params["mu_x"]) / params["sigma2_x"]
        V1 = self._n * np.array(
            [
                [1 / params["sigma2_x"], x12],
                [x12, 0.5 * (vbar - params["mu_x"] ** 2) / params["sigma2_x"]],
            ]
        )

        A = np.diag(self._n_s / params["sigma2_w"])
        B = np.diag(self._n_s * xbar_s / params["sigma2_w"])
        C = np.diag(
            self._n_s
            * (self._wbar_s - params["a"] - params["b"] * xbar_s)
            / params["sigma2_w"]
        )
        D = np.diag(self._n_s * vbar_s / params["sigma2_w"])
        E = np.diag(
            self._n_s
            * (wxbar_s - params["a"] * xbar_s - params["b"] * vbar_s)
            / params["sigma2_w"]
        )
        F = np.diag(
            self._n_s
            * (
                self._wwbar_s
                + params["a"] ** 2
                + params["b"] ** 2 * vbar_s
                - 2 * params["a"] * self._wbar_s
                - 2 * params["b"] * wxbar_s
                + 2 * params["a"] * params["b"] * xbar_s
            )
            * 0.5
            / params["sigma2_w"]
        )
        V2 = np.concatenate(
            [
                np.concatenate([A, B, C], axis=1),
                np.concatenate([B, D, E], axis=1),
                np.concatenate([C, E, F], axis=1),
            ],
            axis=0,
        )

        sigma2_y_long = params["sigma2_y"][self._ind_inv]
        if self._Z is not None:
            B = (self._n * xzbar)[None, :]
            E = self._n_s[None, :] * self._zbar_s / params["sigma2_y"]
            G = (self._Z.T * sigma2_y_long) @ self._Z
            H = (
                self._n_s
                * (
                    self._yzbar_s.T
                    - params["beta_0"] * self._zbar_s.T
                    - params["beta_x"] * xzbar_s.T
                    - (self._zzbar_s @ params["beta_z"]).T
                )
                / params["sigma2_y"]
            )

            dz2 = np.sum(
                params["beta_z"] * params["beta_z"][:, None] * self._zzbar_s,
                axis=(1, 2),
            )
            J_zpart = (
                dz2
                - 2 * self._yzbar_s @ params["beta_z"]
                + 2 * params["beta_0"] * self._zbar_s @ params["beta_z"]
                + 2 * xzbar_s * params["beta_z"] * params["beta_x"]
            )
            C_zpart = -xzbar @ params["beta_z"]
            F_zpart = -self._zbar_s @ params["beta_z"]
        else:
            J_zpart = 0.0
            C_zpart = 0.0
            F_zpart = 0.0
        K = np.array([[np.sum(self._Xhat / sigma2_y_long)]])
        A = (self._n_s * xbar_s / params["sigma2_y"])[None, :]
        C = (
            (
                xybar_s
                - params["beta_0"] * xbar_s
                - params["beta_x"] * vbar_s
                + C_zpart
            )
            / params["sigma2_y"]
        )[None, :]
        D = np.diag(self._n_s / params["sigma2_y"])
        F = np.diag(
            self._n_s
            * (
                self._ybar_s
                - params["beta_0"]
                - params["beta_x"] * xbar_s
                + F_zpart
            )
            / params["sigma2_y"]
        )
        J = np.diag(
            0.5
            * self._n_s
            * (
                self._yybar_s
                + params["beta_0"] ** 2
                + params["beta_x"] ** 2 * vbar_s
                - 2 * params["beta_0"] * xbar_s
                - 2 * params["beta_x"] * xybar_s
                + 2 * params["beta_0"] * params["beta_x"] * xbar_s
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
        pbar: bool = True,
        variance_estimate: bool = False,
        lr: float = 1.0,
        nsample_IS: int = 10000,
    ) -> None:
        super().__init__(
            X,
            S,
            W,
            Y,
            Z,
            max_iter,
            thre,
            max_iter_inner,
            thre_inner,
            pbar,
            variance_estimate,
        )
        self._lr = lr
        self._nIS = nsample_IS

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

    def init(self) -> dict:
        """初始化权重"""
        params = super().init()

        # beta
        Xo_des = [np.ones((self._Xo.shape[0], 1)), self._Xo[:, None]]
        if self._Z is not None:
            Xo_des.append(self._Zo)
        Xo_des = np.concatenate(Xo_des, axis=1)
        beta = logistic(Xo_des, self._Yo)

        params["beta_0"] = np.full(self._ns, beta[0])
        params["beta_x"] = beta[1]
        params["beta_z"] = None if self._Z is None else beta[2:]
        return params

    def e_step(self, params: dict):
        mu_x = params["mu_x"]
        sigma2_x = params["sigma2_x"]
        beta_x = params["beta_x"]
        beta_z = params["beta_z"]
        beta_0_m_long = params["beta_0"][self._ind_m_inv]
        a_m_long = params["a"][self._ind_m_inv]
        b_m_long = params["b"][self._ind_m_inv]
        sigma2_w_m_long = params["sigma2_w"][self._ind_m_inv]

        # 使用newton-raphson方法得到Laplacian approximation
        b_m_long_2 = b_m_long**2
        beta_x_2 = params["beta_x"] ** 2
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

            diff = np.max(np.abs(grad))
            logger.info(f"E step Newton-Raphson: iter={i} diff={diff:.4f}")
            if diff < self._thre_inner:
                break

            Xm -= self._lr * grad / hessian
        else:
            logger.warning(
                f"E step Newton-Raphson (max_iter={self._max_iter_inner})"
                " doesn't converge"
            )
        # 不要使用multivariate_norm，会导致维数灾难，
        # 因为Xm的每个分量都是独立的，使用单变量的norm会好一些
        norm_lap = norm(loc=Xm, scale=1 / np.sqrt(hessian))

        # 进行IS采样
        self._xIS = norm_lap.rvs(size=(self._nIS, self._n_m))  # N x n_m

        # 计算对应的(normalized) importance weights
        pIS = log_expit(
            (2 * self._Ym - 1)
            * (beta_0_m_long + beta_x * self._xIS + Z_part_m)
        )
        pIS -= 0.5 * (
            (self._Wm - a_m_long - b_m_long * self._xIS) ** 2 / sigma2_w_m_long
            + (self._xIS - beta_x) ** 2 / sigma2_x
        )
        pIS = pIS - norm_lap.logpdf(self._xIS)
        self._wIS = softmax(pIS, axis=0)

        if logger.level <= logging.INFO:
            Seff = 1 / np.sum(self._wIS**2, axis=0)
            logger.info(
                "Importance effective size "
                + f"is {Seff.mean():.2f}±{Seff.std():.2f}"
            )

        # 计算Xhat和Xhat2
        self._Xhat[self._is_m] = np.sum(self._xIS * self._wIS, axis=0)
        self._Xhat2[self._is_m] = np.sum(self._xIS**2 * self._wIS, axis=0)

    def m_step(self, params: dict) -> dict:
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
        beta_ = np.r_[params["beta_x"], params["beta_0"]]  # 1+S+p
        if self._Z is not None:
            beta_ = np.r_[beta_, params["beta_z"]]
        Xm_des = [self._xIS[:, :, None], self._Cm]  # 1000 * 360 * 5
        if self._Z is not None:
            Xm_des.append(self._Zm[None, ...])
        Xm_des = np.concatenate(Xm_des, axis=-1)

        beta_ = newton_raphson_beta(
            beta_,
            self._Xo_des,
            Xm_des,
            self._Yo,
            self._Ym,
            self._wIS,
            self._lr,
        )
        beta_x, beta_0 = beta_[0], beta_[1 : (self._ns + 1)]
        beta_z = beta_[(self._ns + 1) :] if self._Z is not None else None

        return {
            "mu_x": mu_x,
            "sigma2_x": sigma2_x,
            "a": a,
            "b": b,
            "sigma2_w": sigma2_w,
            "beta_x": beta_x,
            "beta_0": beta_0,
            "beta_z": beta_z,
        }


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
                pbar=self.pbar_,
            )
        elif self.outcome_type_ == "binary":
            estimator = BinaryEM(
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
            )
        else:
            raise NotImplementedError
        estimator.run()
        self.params_ = estimator.params_
