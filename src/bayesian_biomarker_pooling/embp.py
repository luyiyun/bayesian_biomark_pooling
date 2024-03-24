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


EPS = 1e-5


def batch_dot(mats: ndarray, vecs: ndarray):
    return (mats * vecs[..., None, :]).sum(axis=-1)


def batch_mat_sq(mats: ndarray):
    return mats.swapaxes(-2, -1) @ mats


def batch_nonzero(mask):
    if mask.ndim == 1:
        return np.nonzero(mask)[0]
    else:
        return np.arange(mask.shape[0])[:, None], np.stack(
            [np.nonzero(mask[i])[0] for i in range(mask.shape[0])],
        )


def ols(x_des: ndarray, y: ndarray) -> np.ndarray:
    hat_mat = np.linalg.inv(batch_mat_sq(x_des)) @ x_des.swapaxes(-2, -1)
    return batch_dot(hat_mat, y)


def iter_update_beta(
    batch_mode: bool,
    beta_x: ndarray,
    beta_0: ndarray,
    sigma2_y: ndarray,
    beta_z: ndarray | None,
    xzbar_s: ndarray,
    zbar_s: ndarray,
    zzbar_s: ndarray,
    yzbar_s: ndarray,
    xybar_s: ndarray,
    xbar_s: ndarray,
    vbar_s: ndarray,
    ybar_s: ndarray,
    yybar_s: ndarray,
    n_s: ndarray,
    max_iter: int,
    delta1: float,
    delta2: float,
):
    ns = beta_0.shape[-1]
    beta_all = np.concatenate(
        [beta_x, beta_0] + ([] if beta_z is None else [beta_z]) + [sigma2_y],
        axis=-1,
    )
    if batch_mode:
        beta_all_whole = beta_all.copy()
        remain_bs_ind = np.arange(beta_x.shape[0])
    for i in range(max_iter):
        # 如果batch mode，则会监控一个remain_ind，每次只更新其中的内容，
        # 如果remain_ind数量变成0，则停止更新
        if batch_mode:
            beta_all = beta_all[remain_bs_ind]
            beta_x = beta_x[remain_bs_ind]
            beta_0 = beta_0[remain_bs_ind]
            sigma2_y = sigma2_y[remain_bs_ind]
            xybar_s = xybar_s[remain_bs_ind]
            xbar_s = xbar_s[remain_bs_ind]
            vbar_s = vbar_s[remain_bs_ind]
            ybar_s = ybar_s[remain_bs_ind]
            yybar_s = yybar_s[remain_bs_ind]
            if beta_z is not None:
                beta_z = beta_z[remain_bs_ind]
            if beta_z is not None:
                xzbar_s = xzbar_s[remain_bs_ind]
                zbar_s = zbar_s[remain_bs_ind]
                zzbar_s = zzbar_s[remain_bs_ind]
                yzbar_s = yzbar_s[remain_bs_ind]

        # 关于z的一些项
        if beta_z is not None:
            xzd = batch_dot(xzbar_s, beta_z)
            zd = batch_dot(zbar_s, beta_z)
            if batch_mode:
                dzzd = np.squeeze(
                    beta_z[:, None, None, :]
                    @ zzbar_s
                    @ beta_z[:, None, :, None]
                )
            else:
                dzzd = np.sum(
                    beta_z * zzbar_s * beta_z[..., None],
                    axis=(-2, -1),
                )
            yzd = batch_dot(yzbar_s, beta_z)
            xzd = batch_dot(xzbar_s, beta_z)
        else:
            xzd = yzd = dzzd = zd = xzd = 0.0

        # 为了避免zero variance的问题，需要使用一些通分技巧
        sigma2_y_prod = np.stack(
            [
                np.prod(np.delete(sigma2_y, i, axis=-1), axis=-1)
                for i in range(ns)
            ]
        )
        if batch_mode:
            sigma2_y_prod = sigma2_y_prod.T

        # beta_x
        beta_x_new = (
            n_s * (xybar_s - beta_0 * xbar_s - xzd) * sigma2_y_prod
        ).sum(axis=-1, keepdims=True) / (n_s * vbar_s * sigma2_y_prod).sum(
            axis=-1, keepdims=True
        )
        # beta_0
        beta_0_new = ybar_s - zd - beta_x_new * xbar_s
        # sigma2_y
        sigma2_y_new = (
            yybar_s
            + beta_0_new**2
            + beta_x_new**2 * vbar_s
            + dzzd
            - 2 * beta_0_new * ybar_s
            - 2 * beta_x_new * xybar_s
            - 2 * yzd
            + 2 * beta_0_new * beta_x_new * xbar_s
            + 2 * beta_0_new * zd
            + 2 * beta_x_new * xzd
        )
        # beta_z
        if beta_z is not None:
            # TODO: sigma2_y_new =0 可能会导致问题
            tmp1 = np.linalg.inv(
                np.sum(
                    n_s[..., None, None]  # (nbs,)ns
                    / sigma2_y_new[..., None, None]  # (nbs,)ns
                    * zzbar_s,  # (nbs,)ns,nz,nz
                    axis=-3,
                )
            )
            tmp2 = np.sum(
                (n_s / sigma2_y_new)[..., None]  # (nbs,)ns
                * (
                    yzbar_s  # (nbs,)ns,nz
                    - beta_0_new[..., None] * zbar_s  # (nbs,)ns,nz
                    - beta_x_new[..., None] * xzbar_s  # (nbs,)1
                ),
                axis=-2,
            )
            beta_z_new = batch_dot(tmp1, tmp2)

        # calculate the relative difference
        beta_all_new = np.concatenate(
            [beta_x_new, beta_0_new]
            + ([] if beta_z is None else [beta_z_new])
            + [sigma2_y_new],
            axis=-1,
        )
        rdiff = np.max(
            np.abs(beta_all_new - beta_all) / (np.abs(beta_all) + delta1),
            axis=-1,
        )
        if batch_mode:
            logger_embp.info(
                f"Inner iteration {i+1}: "
                f"relative difference avg is {rdiff.mean(): .4f}"
            )
        else:
            logger_embp.info(
                f"Inner iteration {i+1}: "
                f"relative difference is {rdiff: .4f}"
            )

        # update parameters
        beta_all = beta_all_new
        # beta_x = beta_x_new
        beta_0 = beta_0_new
        sigma2_y = sigma2_y_new
        if beta_z is not None:
            beta_z = beta_z_new
        if batch_mode:
            beta_all_whole[remain_bs_ind] = beta_all_new

        if not batch_mode and (rdiff < delta2):
            logger_embp.info(f"Inner iteration stop, stop iter: {i+1}")
            break
        elif batch_mode:
            remain_bs_ind = np.nonzero(rdiff >= delta2)[0]
            if len(remain_bs_ind) == 0:
                logger_embp.info(f"All inner iteration stop, stop iter: {i+1}")
                break
            else:
                logger_embp.info(
                    f"All inner iteration remain {len(remain_bs_ind)} "
                    "rows to converge"
                )

    else:
        logger_embp.warn("Inner iteration does not converge")

    if batch_mode:
        return beta_all_whole
    return beta_all


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
        assert X.shape == S.shape == W.shape == Y.shape
        if Z is not None:
            assert X.shape == Z.shape[:-1]
        assert X.ndim <= 2

        self._batch_mode = X.ndim > 1
        if self._batch_mode:
            self._n_batch = X.shape[0]

        self._X = X
        self._S = S
        self._W = W
        self._Y = Y
        self._Z = Z

    def prepare(self):

        # 准备后续步骤中会用到的array，预先计算，节省效率
        # NOTE: 默认batch mode下，也不会出现不一致的情况。
        if self._batch_mode:
            st_seq, ind_seq = [], []
            for i in range(self._S.shape[0]):
                uniq_i, ind_inv_i = np.unique(self._S[i], return_inverse=True)
                st_seq.append(uniq_i)
                ind_seq.append(ind_inv_i)
            self._studies, self._ind_inv = np.stack(st_seq), np.stack(ind_seq)
        else:
            self._studies, self._ind_inv = np.unique(
                self._S, return_inverse=True
            )
        self._is_m = pd.isnull(self._X)
        self._is_o = ~self._is_m
        self._ind_m = batch_nonzero(self._is_m)
        self._ind_o = batch_nonzero(self._is_o)

        self._Xo = self._X[self._ind_o]
        self._Yo = self._Y[self._ind_o]
        self._Wo = self._W[self._ind_o]
        self._Ym = self._Y[self._ind_m]
        self._Wm = self._W[self._ind_m]
        if self._Z is not None:
            self._Zo = self._Z[self._ind_o]
            self._Zm = self._Z[self._ind_m]

        if self._batch_mode:
            self._ind_S = [
                batch_nonzero(self._S == s[:, None]) for s in self._studies.T
            ]
            self._ind_Sm = [
                batch_nonzero((self._S == s[:, None]) & self._is_m)
                for s in self._studies.T
            ]
            self._ind_So = [
                batch_nonzero((self._S == s[:, None]) & self._is_o)
                for s in self._studies.T
            ]
            self._ind_m_inv = (
                np.arange(self._is_m.shape[0])[:, None],
                np.stack(
                    [
                        self._ind_inv[i, self._is_m[i]]
                        for i in range(self._is_m.shape[0])
                    ],
                    axis=0,
                ),
            )
        else:
            self._ind_S = [np.nonzero(self._S == s)[0] for s in self._studies]
            self._ind_Sm = [
                np.nonzero((self._S == s) & self._is_m)[0]
                for s in self._studies
            ]
            self._ind_So = [
                np.nonzero((self._S == s) & self._is_o)[0]
                for s in self._studies
            ]
            self._ind_m_inv = self._ind_inv[self._is_m]

        self._n = self._Y.shape[-1]
        self._ns = self._studies.shape[-1]
        self._nz = 0 if self._Z is None else self._Z.shape[-1]
        self._n_o = self._is_o.sum(axis=-1)
        self._n_m = self._is_m.sum(axis=-1)
        self._n_s = np.array(
            [
                indi[1].shape[-1] if self._batch_mode else indi.shape[-1]
                for indi in self._ind_S
            ]
        )
        self._n_ms = np.array(
            [
                indi[1].shape[-1] if self._batch_mode else indi.shape[-1]
                for indi in self._ind_Sm
            ]
        )

        self._wbar_s = np.stack(
            [np.mean(self._W[ind], axis=-1) for ind in self._ind_S]
        )
        self._wwbar_s = np.stack(
            [np.mean(self._W[ind] ** 2, axis=-1) for ind in self._ind_S]
        )
        if self._batch_mode:
            self._wbar_s = self._wbar_s.T
            self._wwbar_s = self._wwbar_s.T

        self._sigma_ind = np.array(
            [1]
            + list(range(2 + 2 * self._ns, 2 + 2 * (self._ns + 1)))
            + list(
                range(3 + 4 * self._ns + self._nz, 3 + 5 * self._ns + self._nz)
            )
        )
        self._params_ind = {
            "mu_x": 0,
            "sigma2_x": 1,
            "a": slice(2, 2 + self._ns),
            "b": slice(2 + self._ns, 2 + 2 * self._ns),
            "sigma2_w": slice(2 + 2 * self._ns, 2 + 3 * self._ns),
        }

        self._Xhat = np.copy(self._X)
        self._Xhat2 = self._Xhat**2

    def init(self) -> ndarray:
        """初始化参数

        这里仅初始化mu_x,sigma2_x,a,b,sigma2_w，其他和outcome相关的参数需要在子类
        中初始化

        Returns:
            dict: 参数组成的dict
        """
        if self._init_method == "reference":
            mu_x = self._Xo.mean(axis=-1, keepdims=True)
            sigma2_x = np.var(self._Xo, ddof=1, axis=-1, keepdims=True)

            a, b, sigma2_w = [], [], []
            for ind_so_i in self._ind_So:
                if len(ind_so_i) == 0:
                    a.append(
                        np.zeros(self._n_batch) if self._batch_mode else 0
                    )
                    b.append(
                        np.zeros(self._n_batch) if self._batch_mode else 0
                    )
                    sigma2_w.append(
                        np.ones(self._n_batch) if self._batch_mode else 1
                    )
                    continue

                Xi, Wi = self._X[ind_so_i], self._W[ind_so_i]
                Xi_des = np.stack([np.ones_like(Xi), Xi], axis=-1)
                abi = ols(Xi_des, Wi)
                sigma2_ws_i = np.mean(
                    (Wi - batch_dot(Xi_des, abi)) ** 2,
                    axis=-1,
                )
                a.append(abi[..., 0])
                b.append(abi[..., 1])
                sigma2_w.append(sigma2_ws_i)
            a, b, sigma2_w = (
                np.stack(a, axis=-1),
                np.stack(b, axis=-1),
                np.stack(sigma2_w, axis=-1),
            )

            res = np.concatenate([mu_x, sigma2_x, a, b, sigma2_w], axis=-1)
            return res
        else:
            raise NotImplementedError
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

    def e_step(self, params: ndarray):
        """Expectation step

        从EM算法的定义上，是计算log joint likelihood的后验期望，也就是Q function。
        但是，在code中一般不是计算这个，而是计算Q function中关于后验期望的部分，
        以便于后面的m step。
        """
        raise NotImplementedError

    def m_step(self, params: ndarray) -> ndarray:
        raise NotImplementedError

    def run(self, init_params: ndarray | None = None):
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

                rdiff = np.max(
                    np.abs(params - params_new)
                    / (np.abs(params) + self._delta1),
                )
                logger_embp.info(
                    f"EM iteration {iter_i}: "
                    f"relative difference is {rdiff: .4f}"
                )
                params = params_new  # 更新
                self.params_hist_.append(params)

                if rdiff < self._delta2:  # TODO: 这里有优化的空间
                    self.iter_convergence_ = iter_i
                    break
            else:
                logger_embp.warning(
                    f"EM iteration (max_iter={self._max_iter}) "
                    "doesn't converge"
                )

        self.params_ = params
        self.params_hist_ = np.stack(self.params_hist_)

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

        self._ybar_s = np.array(
            [self._Y[ind].mean(axis=-1) for ind in self._ind_S]
        )
        self._yybar_s = np.array(
            [(self._Y[ind] ** 2).mean(axis=-1) for ind in self._ind_S]
        )
        if self._batch_mode:
            self._ybar_s = self._ybar_s.T
            self._yybar_s = self._yybar_s.T

        if self._Z is not None:
            self._zzbar_s = []
            self._zbar_s = []
            self._yzbar_s = []
            for ind, n_s in zip(self._ind_S, self._n_s):
                Zs = self._Z[ind]
                self._zzbar_s.append(Zs.swapaxes(-1, -2) @ Zs / n_s)
                self._zbar_s.append(Zs.mean(axis=-2))
                self._yzbar_s.append(
                    (Zs * self._Y[ind][..., None]).mean(axis=-2)
                )
            self._zzbar_s = np.stack(self._zzbar_s, axis=0)
            self._zbar_s = np.stack(self._zbar_s)
            self._yzbar_s = np.stack(self._yzbar_s)
            if self._batch_mode:
                self._zzbar_s = self._zzbar_s.swapaxes(0, 1)
                self._zbar_s = self._zbar_s.swapaxes(0, 1)
                self._yzbar_s = self._yzbar_s.swapaxes(0, 1)
        else:
            self._zzbar_s = self._zbar_s = self._yzbar_s = 0

        self._params_ind.update(
            {
                "beta_x": 2 + 3 * self._ns,
                "beta_0": slice(3 + 3 * self._ns, 3 + 4 * self._ns),
                "beta_z": slice(3 + 4 * self._ns, 3 + 4 * self._ns + self._nz),
                "sigma2_y": slice(
                    3 + 4 * self._ns + self._nz, 3 + 5 * self._ns + self._nz
                ),
            }
        )

    def init(self) -> ndarray:
        params = super().init()

        if self._init_method == "reference":
            Xo_des = [np.ones_like(self._Xo)[..., None], self._Xo[..., None]]
            if self._Z is not None:
                Xo_des.append(self._Zo)
            Xo_des = np.concatenate(Xo_des, axis=-1)
            beta = ols(Xo_des, self._Yo)
            sigma2_ys = np.mean(
                (self._Yo - batch_dot(Xo_des, beta)) ** 2,
                axis=-1,
            )
            if self._batch_mode:
                res = np.concatenate(
                    [
                        params,
                        beta[:, [1] + [0] * self._ns],
                        beta[:, 2:],
                        np.tile(sigma2_ys[:, None], (1, 4)),
                    ],
                    axis=1,
                )
            else:
                res = np.r_[
                    params,
                    beta[1],
                    [beta[0]] * self._ns,
                    beta[2:],
                    [sigma2_ys] * self._ns,
                ]
            return res
        else:
            raise NotImplementedError
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

    def e_step(self, params: ndarray):
        mu_x, sigma2_x, a, b, sigma2_w, beta_x, beta_0, sigma2_y = (
            params[..., self._params_ind[k]]
            for k in [
                "mu_x",
                "sigma2_x",
                "a",
                "b",
                "sigma2_w",
                "beta_x",
                "beta_0",
                "sigma2_y",
            ]
        )
        beta_z = (
            params[..., self._params_ind["beta_z"]]
            if self._Z is not None
            else 0.0
        )

        mu_x = np.expand_dims(mu_x, axis=-1)
        sigma2_x = np.expand_dims(sigma2_x, axis=-1)
        beta_x = np.expand_dims(beta_x, axis=-1)

        sigma2_denominator = (
            sigma2_w * sigma2_x * beta_x**2
            + sigma2_y * sigma2_x * b**2
            + sigma2_y * sigma2_w
        )
        sigma2 = sigma2_w * sigma2_x * sigma2_y / sigma2_denominator

        z_m_part = 0.0 if self._Z is None else batch_dot(self._Zm, beta_z)
        beta_0_m_long = beta_0[self._ind_m_inv]
        sigma2_y_m_long = sigma2_y[self._ind_m_inv]
        a_m_long = a[self._ind_m_inv]
        b_m_long = b[self._ind_m_inv]
        sigma2_w_m_long = sigma2_w[self._ind_m_inv]
        sigma2_denominator_m_long = sigma2_denominator[self._ind_m_inv]
        sigma2_m_long = sigma2[self._ind_m_inv]

        xhat_m = (
            (self._Ym - beta_0_m_long - z_m_part)
            * beta_x
            * sigma2_w_m_long
            * sigma2_x
            + (self._Wm - a_m_long) * b_m_long * sigma2_y_m_long * sigma2_x
            + mu_x * sigma2_w_m_long * sigma2_y_m_long
        ) / sigma2_denominator_m_long

        self._Xhat[self._ind_m] = xhat_m
        self._Xhat2[self._ind_m] = xhat_m**2 + sigma2_m_long

    def m_step(self, params: ndarray) -> ndarray:
        vbar = self._Xhat2.mean(axis=-1)
        wxbar_s = np.stack(
            [
                np.mean(self._W[ind] * self._Xhat[ind], axis=-1)
                for ind in self._ind_S
            ]
        )
        vbar_s = np.stack(
            [np.mean(self._Xhat2[ind], axis=-1) for ind in self._ind_S]
        )
        xbar_s = np.stack(
            [np.mean(self._Xhat[ind], axis=-1) for ind in self._ind_S]
        )
        xybar_s = np.array(
            [
                np.mean(self._Xhat[ind] * self._Y[ind], axis=-1)
                for ind in self._ind_S
            ]
        )
        if self._batch_mode:
            wxbar_s = wxbar_s.T
            vbar_s = vbar_s.T
            xbar_s = xbar_s.T
            xybar_s = xybar_s.T

        if self._Z is not None:
            # xzbar = np.mean(self._Xhat[:, None] * self._Z, axis=0)
            xzbar_s = np.stack(
                [
                    np.mean(self._Xhat[ind][..., None] * self._Z[ind], axis=-2)
                    for ind in self._ind_S
                ],
                axis=0,
            )
            if self._batch_mode:
                xzbar_s = xzbar_s.swapaxes(0, 1)
        else:
            xzbar_s = 0.

        # 3. M step，更新参数值
        mu_x = np.mean(self._Xhat, axis=-1)
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
        beta_all = iter_update_beta(
            batch_mode=self._batch_mode,
            beta_x=params[..., [self._params_ind["beta_x"]]],
            beta_0=params[..., self._params_ind["beta_0"]],
            sigma2_y=params[..., self._params_ind["sigma2_y"]],
            beta_z=(
                None
                if self._Z is None
                else params[..., self._params_ind["beta_z"]]
            ),
            xzbar_s=xzbar_s,
            zbar_s=self._zbar_s,
            zzbar_s=self._zzbar_s,
            yzbar_s=self._yzbar_s,
            xybar_s=xybar_s,
            xbar_s=xbar_s,
            vbar_s=vbar_s,
            ybar_s=self._ybar_s,
            yybar_s=self._yybar_s,
            n_s=self._n_s,
            max_iter=self._max_iter_inner,
            delta1=self._delta1_inner,
            delta2=self._delta2_inner,
        )

        if self._batch_mode:
            return np.concatenate(
                [mu_x[:, None], sigma2_x[:, None], a, b, sigma2_w, beta_all],
                axis=-1,
            )
        else:
            return np.r_[
                mu_x,
                sigma2_x,
                a,
                b,
                sigma2_w,
                beta_all,
            ]

    def v_joint(self, params: ndarray) -> ndarray:
        raise NotImplementedError
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

    init_params = estimator.params_.copy()
    ind_bootstrap = []
    for si in Svals:
        for is_i in [is_m, not_m]:
            if Y_type == "continue":
                ind = np.nonzero((S == si) & is_i)[0]
                if len(ind) == 0:
                    continue
                ind_choice = seed.choice(
                    ind, (n_repeat, len(ind)), replace=True
                )
                ind_bootstrap.append(ind_choice)
            elif Y_type == "binary":
                for yi in Yvals:
                    ind = np.nonzero((S == si) & (Y == yi) & is_i)[0]
                    if len(ind) == 0:
                        continue
                    ind_choice = seed.choice(
                        ind, (n_repeat, len(ind)), replace=True
                    )
                    ind_bootstrap.append(ind_choice)
            else:
                raise NotImplementedError
    ind_bootstrap = np.concatenate(ind_bootstrap, axis=1)  # nbs x N
    X_bs, Y_bs, W_bs, S_bs, Z_bs = (
        X[ind_bootstrap],  # nbs x N
        Y[ind_bootstrap],
        W[ind_bootstrap],
        S[ind_bootstrap],
        None if Z is None else Z[ind_bootstrap],  # nbs x N x nz
    )
    estimator.register_data(X_bs, S_bs, W_bs, Y_bs, Z_bs)
    # 使用EM估计值作为初始值
    estimator.run(init_params=np.tile(init_params[None, :], (n_repeat, 1)))

    return estimator.params_


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

        params_names = []
        for k, v in self._estimator._params_ind.items():
            if isinstance(v, slice):
                params_names.extend([k] * (v.stop - v.start))
            else:
                params_names.append(k)
        self.params_ = pd.DataFrame(
            {"estimate": self._estimator.params_}, index=params_names
        )
        self.params_hist_ = pd.DataFrame(
            self._estimator.params_hist_, columns=params_names
        )

        if not self.var_est_:
            return

        if self.var_est_method_ == "bootstrap":
            # 使用boostrap方法
            if self.pbar_:
                print("Bootstrap: ")
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
            )
            res_ci = np.quantile(
                res_bootstrap,
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
