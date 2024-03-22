import logging

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from torch import Tensor
from numpy import ndarray

from .logger import logger_embp


def ols(x_des: Tensor, y: Tensor) -> Tensor:
    return torch.inverse(x_des.T @ x_des) @ x_des.T @ y


def logistic(
    Xdes: Tensor,
    y: Tensor,
    lr: float = 1.0,
    max_iter: int = 100,
    delta1: float = 1e-3,
    delta2: float = 1e-4,
):
    beta = torch.zeros(Xdes.shape[1]).to(Xdes)

    for i in range(max_iter):
        p = torch.sigmoid(Xdes @ beta)
        grad = Xdes.T @ (p - y)
        hessian = (Xdes.T * p * (1 - p)) @ Xdes
        delta = lr * torch.inverse(hessian) @ grad
        beta -= delta

        rdiff = (delta.abs() / (beta.abs() + delta1)).max().item()
        logger_embp.info(f"Init Newton-Raphson: iter={i+1} diff={rdiff:.4f}")
        if rdiff < delta2:
            break
    else:
        logger_embp.warning(
            f"Init Newton-Raphson (max_iter={max_iter}) doesn't converge"
        )

    return beta


def newton_raphson_beta(
    init_beta: Tensor,
    Xo_des: Tensor,  # ns x (1+S+p)
    Xm_des: Tensor,  # N x nm x (1+S+p)
    Yo: Tensor,
    Ym: Tensor,
    wIS: Tensor,  # N x nm
    lr: float = 1.0,
    max_iter: int = 100,
    delta1: float = 1e-3,
    delta2: float = 1e-4,
):
    beta_ = init_beta
    for i in range(max_iter):
        p_o = torch.sigmoid(Xo_des @ beta_)  # ns
        p_m = torch.sigmoid(Xm_des @ beta_)  # N x nm
        mul_o = p_o - Yo  # ns
        mul_m = (p_m - Ym) * wIS  # N x nm
        grad = Xo_des.T @ mul_o + (Xm_des * mul_m[..., None]).sum(dim=(0, 1))

        H_o = (Xo_des.T * p_o * (1 - p_o)) @ Xo_des
        H_m = torch.sum(
            (Xm_des * (wIS * p_m * (1 - p_m))[..., None]).transpose(1, 2)
            @ Xm_des,
            dim=0,
        )
        H = H_o + H_m

        beta_delta = lr * torch.inverse(H) @ grad
        beta_ -= beta_delta

        rdiff = (beta_delta.abs() / (beta_.abs() + delta1)).max()
        logger_embp.info(f"M step Newton-Raphson: iter={i+1} diff={rdiff:.4f}")
        if rdiff < delta2:
            break
    else:
        logger_embp.warning(
            f"M step Newton-Raphson (max_iter={max_iter}) doesn't converge"
        )

    return beta_


class EMTorch:

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
        pbar: bool = True,
        device: str = "cuda:0",
    ) -> None:
        self._device = torch.device(device)

        self._max_iter = max_iter
        self._max_iter_inner = max_iter_inner
        self._delta1 = delta1
        self._delta1_inner = delta1_inner
        self._delta2 = delta2
        self._delta2_inner = delta2_inner
        self._delta1_var = delta1_var
        self._delta2_var = delta2_var
        self._pbar = pbar

    def register_data(
        self,
        X: ndarray,
        S: ndarray,
        W: ndarray,
        Y: ndarray,
        Z: ndarray | None = None,
    ):
        self._X = torch.tensor(X, device=self._device, dtype=torch.float64)
        self._S = torch.tensor(S, device=self._device, dtype=torch.float64)
        self._W = torch.tensor(W, device=self._device, dtype=torch.float64)
        self._Y = torch.tensor(Y, device=self._device, dtype=torch.float64)
        if Z is not None:
            self._Z = torch.tensor(Z, device=self._device, dtype=torch.float64)
        else:
            self._Z = None

    def prepare(self):
        # 准备后续步骤中会用到的array，预先计算，节省效率
        self._n = self._Y.shape[0]
        self._studies, self._ind_inv = torch.unique(
            self._S, return_inverse=True
        )
        self._is_m = torch.isnan(self._X)
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

        self._ind_S = [
            torch.nonzero(self._S == s).squeeze() for s in self._studies
        ]
        self._ind_Sm = [
            torch.nonzero((self._S == s) & self._is_m).squeeze()
            for s in self._studies
        ]
        self._ind_So = [
            torch.nonzero((self._S == s) & self._is_o).squeeze()
            for s in self._studies
        ]
        self._ind_m_inv = self._ind_inv[self._is_m]

        self._n_s = torch.tensor([len(indi) for indi in self._ind_S])
        self._n_s.to(self._Y)
        self._n_ms = torch.tensor([len(indi) for indi in self._ind_Sm])
        self._n_ms.to(self._Y)

        self._wbar_s = torch.stack(
            [self._W[ind].mean() for ind in self._ind_S]
        )
        self._wwbar_s = torch.stack(
            [(self._W[ind] ** 2).mean() for ind in self._ind_S]
        )

        self._Xhat = self._X.clone()
        self._Xhat2 = self._Xhat**2

    def init(self) -> dict[str, Tensor]:
        """初始化参数

        这里仅初始化mu_x,sigma2_x,a,b,sigma2_w，其他和outcome相关的参数需要在子类
        中初始化
        """
        mu_x = self._Xo.mean()
        sigma2_x = self._Xo.var()

        a, b, sigma2_w = [], [], []
        for ind_so_i in self._ind_So:
            if len(ind_so_i) == 0:
                a.append(0)
                b.append(0)
                sigma2_w.append(1)
                continue

            Xi, Wi = self._X[ind_so_i], self._W[ind_so_i]
            Xi_des = torch.stack([torch.ones(Xi.shape[0]).to(Xi), Xi], dim=1)
            abi = ols(Xi_des, Wi)
            sigma2_ws_i = torch.mean((Wi - Xi_des @ abi) ** 2)
            a.append(abi[0])
            b.append(abi[1])
            sigma2_w.append(sigma2_ws_i)

        a, b, sigma2_w = torch.stack(a), torch.stack(b), torch.stack(sigma2_w)
        return {
            "mu_x": mu_x,
            "sigma2_x": sigma2_x,
            "a": a,
            "b": b,
            "sigma2_w": sigma2_w,
        }

    def e_step(self, params: dict[str, Tensor]):
        """Expectation step

        从EM算法的定义上，是计算log joint likelihood的后验期望，也就是Q function。
        但是，在code中一般不是计算这个，而是计算Q function中关于后验期望的部分，
        以便于后面的m step。
        """
        raise NotImplementedError

    def m_step(self, params: dict[str, Tensor]) -> pd.Series:
        raise NotImplementedError

    def run(self):

        def to_series(params: dict[str, Tensor]):
            names, values = [], []
            for k in [
                "mu_x",
                "sigma2_x",
                "a",
                "b",
                "sigma2_w",
                "beta_x",
                "beta_0",
                "beta_z",
            ]:
                if k not in params:
                    continue
                tensor = params[k]
                if tensor.ndim == 0:
                    names.append(k)
                    tensor = tensor.unsqueeze(0)
                else:
                    names.extend([k] * tensor.shape[0])
                values.append(tensor)
            return pd.Series(torch.cat(values).cpu().numpy(), index=names)

        self.prepare()

        params = self.init()
        self.params_hist_ = [to_series(params)]
        with logging_redirect_tqdm(loggers=[logger_embp]):
            for iter_i in tqdm(
                range(1, self._max_iter + 1),
                desc="EM: ",
                disable=not self._pbar,
            ):

                self.e_step(params)
                params_new = self.m_step(params)

                params_old_ser = to_series(params)
                params_new_ser = to_series(params_new)
                rdiff = (
                    (params_old_ser - params_new_ser).abs()
                    / (params_old_ser.abs() + self._delta1)
                ).max()
                logger_embp.info(
                    f"EM iteration {iter_i}: difference is {rdiff: .4f}"
                )
                params = params_new  # 更新
                self.params_hist_.append(params_new_ser)
                if rdiff < self._delta2:
                    self.iter_convergence_ = iter_i
                    break
            # else:
            #     logger_embp.warning(
            #         f"EM iteration (max_iter={self._max_iter}) "
            #         "doesn't converge"
            #     )

        self.params_ = params_new_ser
        self.params_hist_ = pd.concat(self.params_hist_, axis=1).T

    def v_joint(self, params: pd.Series) -> ndarray:
        raise NotImplementedError

    def estimate_variance(self) -> ndarray:
        raise NotImplementedError
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


class BinaryEMTorch(EMTorch):

    def __init__(
        self,
        max_iter: int = 100,
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
        device: str = "cuda:0",
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
            device=device,
        )
        self._lr = lr
        self._nIS = n_importance_sampling

    def prepare(self):
        super().prepare()

        C = torch.zeros((self._n, self._ns)).to(self._Y)
        for i in range(self._ns):
            C[self._ind_S[i], i] = 1
        self._Co = C[self._is_o, :]
        self._Cm = torch.tile(C[None, self._is_m, :], (self._nIS, 1, 1))

        Xo_des = [self._Xo[:, None], self._Co]  # 360 * 5
        if self._Z is not None:
            Xo_des.append(self._Zo)
        self._Xo_des = torch.concat(Xo_des, dim=-1)

    def init(self) -> dict[str, Tensor]:
        """初始化权重"""
        params = super().init()

        # beta
        Xo_des = [
            torch.ones((self._Xo.shape[0], 1)).to(self._Xo),
            self._Xo[:, None],
        ]
        if self._Z is not None:
            Xo_des.append(self._Zo)
        Xo_des = torch.cat(Xo_des, dim=1)
        beta = logistic(
            Xo_des,
            self._Yo,
            lr=self._lr,
            max_iter=self._max_iter_inner,
            delta1=self._delta1_inner,
            delta2=self._delta2_inner,
        )

        params_update = {
            "beta_x": beta[1],
            "beta_0": torch.tile(beta[0], (self._ns,)),
        }
        if self._Z is not None:
            params_update["beta_z"] = beta[2:]
        params.update(params_update)
        return params

    def e_step(self, params: dict[str, Tensor]):
        mu_x = params["mu_x"]
        sigma2_x = params["sigma2_x"]
        a = params["a"]
        b = params["b"]
        sigma2_w = params["sigma2_w"]
        beta_x = params["beta_x"]
        beta_z = params["beta_z"] if self._Z is not None else 0.0
        beta_0 = params["beta_0"]

        beta_0_m_long = beta_0[self._ind_m_inv]
        a_m_long = a[self._ind_m_inv]
        b_m_long = b[self._ind_m_inv]
        sigma2_w_m_long = sigma2_w[self._ind_m_inv]

        # 使用newton-raphson方法得到Laplacian approximation
        Xm = 0  # np.random.randn(Wm.shape[0])
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
            p = torch.sigmoid(Xm * beta_x + delta_part)
            grad = beta_x * p + grad_mul * Xm + grad_const
            hessian = beta_x_2 * p * (1 - p) + grad_mul

            xdelta = self._lr * grad / hessian
            Xm -= xdelta

            rdiff = (xdelta.abs() / (Xm.abs() + self._delta1_inner)).max()
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
        p = torch.sigmoid(Xm * beta_x + delta_part)
        hessian = beta_x_2 * p * (1 - p) + grad_mul

        # 不要使用multivariate_norm，会导致维数灾难，
        # 因为Xm的每个分量都是独立的，使用单变量的norm会好一些
        norm_lap = torch.distributions.Normal(loc=Xm, scale=1 / hessian.sqrt())

        # 进行IS采样
        self._XIS = norm_lap.sample((self._nIS,))  # N x n_m

        # 计算对应的(normalized) importance weights
        pIS = torch.nn.functional.logsigmoid(
            (2 * self._Ym - 1)
            * (beta_0_m_long + beta_x * self._XIS + Z_part_m)
        )
        pIS -= 0.5 * (
            (self._Wm - a_m_long - b_m_long * self._XIS) ** 2 / sigma2_w_m_long
            + (self._XIS - mu_x) ** 2 / sigma2_x
        )
        pIS = pIS - norm_lap.log_prob(self._XIS)
        self._WIS = torch.softmax(pIS, dim=0)

        if logger_embp.level <= logging.INFO:
            Seff = 1 / torch.sum(self._WIS**2, dim=0)
            logger_embp.info(
                "Importance effective size "
                + f"is {Seff.mean():.2f}±{Seff.std():.2f}"
            )

        # 计算Xhat和Xhat2
        self._Xhat[self._is_m] = torch.sum(self._XIS * self._WIS, dim=0)
        self._Xhat2[self._is_m] = torch.sum(self._XIS**2 * self._WIS, dim=0)

    def m_step(self, params: dict[str, Tensor]) -> dict[str, Tensor]:
        beta_x = params["beta_x"]
        beta_z = (
            params["beta_z"]
            if self._Z is not None
            else torch.tensor([]).to(beta_x)
        )
        beta_0 = params["beta_0"]

        vbar = self._Xhat2.mean()
        xbars = torch.stack([self._Xhat[sind].mean() for sind in self._ind_S])
        vbars = torch.stack([self._Xhat2[sind].mean() for sind in self._ind_S])
        wxbars = torch.stack(
            [(self._W[sind] * self._Xhat[sind]).mean() for sind in self._ind_S]
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
        beta_ = torch.cat([beta_x.unsqueeze(0), beta_0, beta_z])
        Xm_des = [self._XIS[:, :, None], self._Cm]
        if self._Z is not None:
            Xm_des.append(self._Zm[None, ...])
        Xm_des = torch.cat(Xm_des, dim=-1)  # N x nm x (1+S+p)

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

        params = {
            "mu_x": mu_x,
            "sigma2_x": sigma2_x,
            "a": a,
            "b": b,
            "sigma2_w": sigma2_w,
            "beta_x": beta_x,
            "beta_0": beta_0,
        }
        if self._Z is not None:
            params["beta_z"] = beta_z
        return params