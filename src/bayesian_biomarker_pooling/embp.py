import logging

import numpy as np

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


def init_ols(Xo: ndarray, Yo: ndarray, Zo: ndarray | None, XWYZ_xKnow: list):
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


class EMBP(BiomarkerPoolBase):

    def __init__(
        self,
        thre: float = 1e-5,
        max_iter: int = 1000,
        thre_inner: float = 1e-7,
        max_iter_inner: int = 100,
    ) -> None:
        self.thre_ = thre
        self.max_iter_ = max_iter
        self.thre_inner_ = thre_inner
        self.max_iter_inner_ = max_iter_inner

    def fit(
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
        params = init_ols(
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
