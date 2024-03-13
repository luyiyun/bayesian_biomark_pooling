import logging
from typing import Literal, List

import pandas as pd
import numpy as np
from scipy.special import expit, log_expit, softmax
from scipy.stats import norm, rv_continuous, multivariate_normal

# import pandas as pd
from numpy import ndarray
from .base import BiomarkerPoolBase, check_split_data


logger = logging.getLogger("EMBP")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "[%(name)s][%(levelname)s][%(asctime)s]:%(message)s"
)
ch.setFormatter(formatter)
logger.addHandler(ch)


def ols(x_des, y) -> np.ndarray:
    return np.linalg.inv(x_des.T @ x_des) @ x_des.T @ y


def init_continue(
    Xo: ndarray, Yo: ndarray, Zo: ndarray | None, XWYZ_xKnow: list
):
    mu_x = Xo.mean()
    sigma2_x = np.var(Xo, ddof=1)

    Xo_des = [np.ones((Xo.shape[0], 1)), Xo[:, None]]
    if Zo is not None:
        Xo_des.append(Zo)
    Xo_des = np.concatenate(Xo_des, axis=1)
    beta = ols(Xo_des, Yo)
    sigma2_ys = np.mean((Yo - Xo_des @ beta) ** 2)

    a, b, sigma2_w = [], [], []
    for item in XWYZ_xKnow:
        if item is None:
            a.append(0)
            b.append(0)
            sigma2_w.append(1)
            continue
        Xi, Wi = item[1:3]
        Xi_des = np.stack([np.ones(Xi.shape[0]), Xi], axis=1)
        abi = ols(Xi_des, Wi)
        sigma2_ws_i = np.mean((Wi - Xi_des @ abi) ** 2)
        a.append(abi[0])
        b.append(abi[1])
        sigma2_w.append(sigma2_ws_i)

    ns = len(XWYZ_xKnow)

    params = {
        "mu_x": mu_x,
        "sigma2_x": sigma2_x,
        "beta_0": np.full(ns, beta[0]),
        "beta_x": beta[1],
        "sigma2_y": np.full(ns, sigma2_ys),
        "a": np.array(a),
        "b": np.array(b),
        "sigma2_w": np.array(sigma2_w),
    }
    if Zo is not None:
        params["beta_z"] = beta[2:]

    return params


def calc_xhat(params: dict, WYZ_xUnKnow: list) -> dict:
    sigma2 = 1 / (
        params["beta_x"] ** 2 / params["sigma2_y"]
        + params["b"] ** 2 / params["sigma2_w"]
        + 1 / params["sigma2_x"]
    )

    xhat, xhat2 = [], []
    for i, item in enumerate(WYZ_xUnKnow):
        if item is None:
            xhat.append(np.array([]))
            xhat2.append(np.array([]))
            continue

        Wi, Yi, Zi = item[1:]
        residual_yi = Yi - params["beta_0"][i]
        if Zi is not None:
            residual_yi -= Zi @ params["beta_z"]
        e_s = (
            residual_yi * params["beta_x"] / params["sigma2_y"][i]
            + (Wi - params["a"][i]) * params["b"][i] / params["sigma2_w"][i]
            + params["mu_x"] / params["sigma2_x"]
        )
        xhat.append(e_s * sigma2[i])
        xhat2.append((e_s * sigma2[i]) ** 2 + sigma2[i])

    return {"sigma2": sigma2, "xhat": xhat, "xhat2": xhat2}


def iter_calc_params(
    Y,
    Z,
    Xhat,
    Xhat2,
    sigma2,
    ind_s,
    ind_s_inv,
    n_ms,
    n_s,
    beta_x,
    beta_0,
    beta_z,
    sigma2_y,
    thre: float = 1e-5,
    max_iter: int = 100,
):

    for iter_i in range(max_iter):
        # update beta_x
        beta_0_long = beta_0[ind_s_inv]
        sigma2_y_long = sigma2_y[ind_s_inv]
        z_part = Z @ beta_z if Z is not None else 0
        resid_betax = (Y - beta_0_long - z_part) * Xhat / sigma2_y_long
        beta_x_new = resid_betax.mean() / (Xhat2 / sigma2_y_long).mean()
        resid_beta0 = Y - z_part - beta_x_new * Xhat
        beta_0_new = np.array([resid_beta0[indi].mean() for indi in ind_s])

        if Z is not None:
            resid_z = (
                Y - beta_0_new[ind_s_inv] - beta_x_new * Xhat
            ) / sigma2_y_long
            beta_z_new = (
                np.linalg.inv(Z.T @ np.diag(sigma2_y_long) @ Z) @ Z.T @ resid_z
            )
            z_part = Z @ beta_z_new

        resid_sigma_2 = (
            Y - beta_0_new[ind_s_inv] - beta_x_new * Xhat - z_part
        ) ** 2
        sigma2_y_new = np.array(
            [resid_sigma_2[ind_si].mean() for ind_si in ind_s]
        )
        sigma2_y_new += n_ms * beta_x_new**2 * sigma2 / n_s

        diff = np.r_[
            beta_x_new - beta_x, beta_0_new - beta_0, sigma2_y_new - sigma2_y
        ]
        if Z is not None:
            diff = np.r_[diff, beta_z_new - beta_z]
        diff = np.max(np.abs(diff))
        logger.info(f"Inner iteration {iter_i+1}: difference is {diff: .4f}")

        beta_x = beta_x_new
        beta_0 = beta_0_new
        sigma2_y = sigma2_y_new
        if Z is not None:
            beta_z = beta_z_new

        if diff < thre:
            logger.info(f"Inner iteration stop: difference is {diff: .4f}")
            break

    return beta_x, beta_0, sigma2_y, beta_z


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
    ) -> None:
        assert outcome_type in ["continue", "binary"]

        self.outcome_type_ = outcome_type
        self.thre_ = thre
        self.max_iter_ = max_iter
        self.thre_inner_ = thre_inner
        self.max_iter_inner_ = max_iter_inner
        self.nsample_IS_ = nsample_IS
        self.lr_ = lr

    def fit(
        self,
        X: ndarray,
        S: ndarray,
        W: ndarray,
        Y: ndarray,
        Z: ndarray | None = None,
    ) -> None:
        if self.outcome_type_ == "continue":
            self._fit_continue(X, S, W, Y, Z)
        elif self.outcome_type_ == "binary":
            self._fit_binary(X, S, W, Y, Z)
        else:
            raise NotImplementedError

    def _fit_continue(
        self,
        X: ndarray,
        S: ndarray,
        W: ndarray,
        Y: ndarray,
        Z: ndarray | None = None,
    ) -> None:
        n = X.shape[0]
        dats = check_split_data(X, S, W, Y, Z)
        ind_o = dats["ind_o"]
        ind_s = dats["ind_s"]
        ind_s_inv = dats["ind_s_inv"]
        n_ms, n_os = dats["n_ms"], dats["n_os"]
        n_s = n_ms + n_os

        Xhat = np.copy(X)
        Xhat2 = Xhat**2

        # 1. 使用OLS得到初始值
        params = init_continue(
            X[ind_o],
            Y[ind_o],
            None if Z is None else Z[ind_o, :],
            dats["XWYZ_xKnow"],
        )

        for iter_i in range(self.max_iter_):

            # 2. E step, 计算hat x
            E_res = calc_xhat(params, dats["WYZ_xUnKnow"])
            sigma2 = E_res["sigma2"]
            for xhat_m, xhat2_m, item in zip(
                E_res["xhat"], E_res["xhat2"], dats["WYZ_xUnKnow"]
            ):
                if item is None:
                    continue
                ind_m_s = item[0]
                Xhat[ind_m_s] = xhat_m
                Xhat2[ind_m_s] = xhat2_m

            # 3. M step，更新参数值
            mu_x = np.mean(Xhat)
            sigma2_x = np.mean((Xhat - mu_x) ** 2) + np.sum(sigma2 * n_s / n)
            a, b, sigma2_w = [], [], []
            for i, ind_si in enumerate(ind_s):
                wsi, xsi, x2si = W[ind_si], Xhat[ind_si], Xhat2[ind_si]
                wsi_ = wsi.mean()
                xsi_ = xsi.mean()
                x2si_ = x2si.mean()

                bsi = ((wsi * xsi).mean() - wsi_ * xsi_) / (x2si_ - xsi_**2)
                resid = wsi - bsi * xsi
                asi = np.mean(resid)
                sigma2_wsi = np.mean((resid - asi) ** 2)
                b.append(bsi)
                a.append(asi)
                sigma2_w.append(sigma2_wsi)
            a, b, sigma2_w = np.array(a), np.array(b), np.array(sigma2_w)
            sigma2_w += b**2 * sigma2 * n_ms / n_s

            # 迭代法求解剩余的参数
            beta_x, beta_0, sigma2_y, beta_z = iter_calc_params(
                Y=Y,
                Z=Z,
                Xhat=Xhat,
                Xhat2=Xhat2,
                sigma2=sigma2,
                ind_s=ind_s,
                ind_s_inv=ind_s_inv,
                n_ms=n_ms,
                n_s=n_s,
                beta_x=params["beta_x"],
                beta_0=params["beta_0"],
                beta_z=params.get("beta_z", 0),
                sigma2_y=params["sigma2_y"],
                thre=self.thre_inner_,
                max_iter=self.max_iter_inner_,
            )

            # 4. 检查是否收敛
            diff = np.r_[
                mu_x - params["mu_x"],
                sigma2_x - params["sigma2_x"],
                beta_x - params["beta_x"],
                beta_0 - params["beta_0"],
                sigma2_y - params["sigma2_y"],
                a - params["a"],
                b - params["b"],
                sigma2_w - params["sigma2_w"],
            ]
            if Z is not None:
                diff = np.r_[diff, beta_z - params["beta_z"]]
            diff = np.max(np.abs(diff))
            logger.info(
                f"Outer iteration {iter_i+1}: difference is {diff: .4f}"
            )

            params["mu_x"] = mu_x
            params["sigma2_x"] = sigma2_x
            params["beta_x"] = beta_x
            params["beta_0"] = beta_0
            params["sigma2_y"] = sigma2_y
            params["a"] = a
            params["b"] = b
            params["sigma2_w"] = sigma2_w
            if Z is not None:
                params["beta_z"] = beta_z

            if diff < self.thre_:
                logger.info(f"Outer iteration stop: difference is {diff: .4f}")
                break

        self.params_ = params

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
