import logging
import os
import hashlib
from collections import abc
from typing import Literal, Optional, Sequence, Union, Tuple, Dict

import numpy as np
import pandas as pd
from scipy import integrate, optimize, special, stats


logger = logging.getLogger("main.simulate")


def prepare_params_source_heterogeneity(
    *params: Union[float, Sequence[float]],
) -> Tuple[Union[None, np.ndarray]]:
    # 通过某些参数的数量，来确定studies的数量
    # 同时需要保证这些参数的数量是一致的
    #  (要么是1, 表示在所有studies中一样, 要么是相同的长度)
    ns = None
    for parami in params:
        if params is None or isinstance(parami, (int, float)):
            continue
        len_parami = len(parami)
        if ns is None:
            ns = len_parami
        else:
            assert ns == len_parami, (
                "the length of a, b, sigma2_e, "
                "beta0, beta1 must be one or equal."
            )
    ns = 1 if ns is None else ns  # 如果全是scalar，则代表是只有一个source

    res = []
    for parami in params:
        if parami is None:
            res.append(None)
        elif isinstance(parami, (float, int)):
            res.append(np.array([parami] * ns))
        else:
            res.append(np.array(parami))
    return tuple(res)


def get_beta0_by_prevalence(
    prevalence: float, beta1: float, mu_x: float, sigma2_x: float
) -> float:
    def p_Y(beta0, beta1, mu_x, sigma_x):
        return integrate.quad(
            lambda x: stats.norm.pdf(x, mu_x, sigma_x)
            * special.expit(beta0 + beta1 * x),
            -np.inf,
            np.inf,
        )[0]

    res = optimize.root_scalar(
        lambda x: p_Y(x, beta1, mu_x, np.sqrt(sigma2_x)) - prevalence, x0=0.0
    )
    return res.root


def hash_dict(d: Dict) -> str:
    s = ",".join(
        [f"{k}:{v}" for k, v in sorted(d.items(), key=lambda x: x[0])]
    )
    return hashlib.md5(s.encode("utf-8")).hexdigest()


class Simulator:
    """可以为每个studies指定不同的样本量、可观测X样本量等。"""

    def __init__(
        self,
        # parameters
        beta_x: float = 1.0,
        beta_z: Optional[Union[float, Sequence[float]]] = None,
        beta_0: Union[float, Sequence[float]] = 1.0,
        prevalence: Optional[Union[float, Sequence[float]]] = None,
        a: Sequence[float] = [-3, 1, -1, 3],
        b: Sequence[float] = [0.5, 0.75, 1.25, 1.5],
        sigma2_e: Union[float, Sequence[float]] = 1.0,
        sigma2_y: Union[float, Sequence[float]] = 1.0,
        # distributions
        mu_x: float = 0.0,
        sigma2_x: float = 1.0,
        mu_z: Union[float, Sequence[float]] = 0.0,
        sigma_z: Union[float, Sequence[float]] = 1.0,
        # number of samples
        n_sample_per_studies: Union[int, Sequence[int]] = 100,
        n_knowX_per_studies: Union[int, Sequence[int]] = 10,
        n_knowX_balance: bool = False,
        # others
        direction: Literal["x->w", "w->x"] = "x->w",
        type_outcome: Literal["binary", "continue"] = "binary",
    ) -> None:
        """
        n_knowX_balance: 保证存在x的样本中y=1 / y=0的比例与所有样本y=1 / y=0的比例相同
        """
        assert direction in ["x->w", "w->x"]
        assert type_outcome in ["binary", "continue"]
        if type_outcome != "binary":
            # 如果不是binary, 则只能直接指定beta0,而不是通过prevalence来间接确定beta0
            assert beta_0 is not None
            assert not n_knowX_balance, "n_knowX_balance=True only when binary"
        if beta_z is not None and isinstance(mu_z, abc.Sequence):
            assert len(beta_z) == len(mu_z)
        if beta_z is not None and isinstance(sigma_z, abc.Sequence):
            assert len(beta_z) == len(mu_z)

        # 如果指定了prevalence，则使用prevalence来计算beta0，
        # prevalence也可以指定多个，来对应多个beta0
        if type_outcome == "binary" and prevalence is not None:
            if isinstance(prevalence, float):
                # compute the suitable beta0 to produce necessary prevalence
                beta_0 = get_beta0_by_prevalence(
                    prevalence, beta_x, mu_x, sigma2_x
                )
                logger.info(
                    f"(pid:{os.getpid()})Get the beta0 = {beta_0:.4f} "
                    f"by prevalence {prevalence:.4f}"
                )
            elif isinstance(prevalence, abc.Sequence):
                beta_0 = []
                for pi in prevalence:
                    beta_0i = get_beta0_by_prevalence(
                        pi, beta_x, mu_x, sigma2_x
                    )
                    beta_0.append(beta_0i)
                logger.info(
                    f"(pid:{os.getpid()})Get the beta0 = {str(beta_0)} "
                    f"by prevalence {str(prevalence)}"
                )

        # 通过某些参数的数量，来确定studies的数量
        # 同时需要保证这些参数的数量是一致的
        #  (要么是1, 表示在所有studies中一样, 要么是相同的长度)
        (
            beta_0,
            a,
            b,
            sigma2_e,
            sigma2_y,
            n_sample_per_studies,
            n_knowX_per_studies,
        ) = prepare_params_source_heterogeneity(
            beta_0,
            a,
            b,
            sigma2_e,
            sigma2_y,
            n_sample_per_studies,
            n_knowX_per_studies,
        )

        self._parameters = {
            "mu_x": mu_x,
            "sigma2_x": sigma2_x,
            "sigma_x": np.sqrt(sigma2_x),
            "sigma2_y": sigma2_y,
            "sigma_y": np.sqrt(sigma2_y),
            "beta_x": beta_x,
            "beta_0": beta_0,
            "beta_z": beta_z,
            "a": a,
            "b": b,
            "sigma2_e": sigma2_e,
            "sigma_e": np.sqrt(sigma2_e),
            "mu_z": mu_z,
            "sigma_z": sigma_z,
            "n_sample_per_studies": n_sample_per_studies,
            "n_knowX_per_studies": n_knowX_per_studies,
            "n_studies": len(beta_0),
            "direction": direction,
            "n_samples": np.sum(n_sample_per_studies),
            "n_knowX_balance": n_knowX_balance,
            "type_outcome": type_outcome,
        }
        self._typ_outcome = type_outcome

        if direction == "w->x":
            mu_w = (mu_x - a) / b
            sigma2_w = (sigma2_x - sigma2_e) / (b**2)

            self._parameters.update(
                {
                    "mu_w": mu_w,
                    "sigma2_w": sigma2_w,
                    "sigma_w": np.sqrt(sigma2_w),
                }
            )

        # self._name = f"prev{prevalence:.2f}-betax{beta_x:.2f}"
        self._name = hash_dict(self._parameters)

    @property
    def parameters(self):
        return self._parameters

    @property
    def parameters_series(self):
        keys = [
            "mu_x",
            "sigma2_x",
            "a",
            "b",
            "sigma2_e",
            "beta_x",
            "beta_0",
            "beta_z",
        ]
        if self._typ_outcome == "continue":
            keys.append("sigma2_y")
        index, res = [], []
        for key in keys:
            val = self._parameters[key]
            if val is None:
                continue
            elif isinstance(val, np.ndarray):
                val = val.tolist()
                res.extend(val)
                index.extend([f"{key}-{i}" for i in range(len(val))])
            else:
                res.append(val)
                index.append(key)

        return pd.Series(res, index=index)

    @property
    def name(self):
        return self._name

    def simulate(self, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)

        Ns = self._parameters["n_sample_per_studies"]
        nKnowX = self._parameters["n_knowX_per_studies"]

        a = np.repeat(self._parameters["a"], Ns)
        b = np.repeat(self._parameters["b"], Ns)
        sigma_e = np.repeat(self._parameters["sigma_e"], Ns)
        e = rng.normal(0, sigma_e)
        if self._parameters["direction"] == "w->x":
            mu_w = np.repeat(self._parameters["mu_w"], Ns)
            sigma_w = np.repeat(self._parameters["sigma_w"], Ns)
            W = rng.normal(mu_w, sigma_w)
            X = a + b * W + e
        else:  # x->w
            X = rng.normal(
                self._parameters["mu_x"],
                self._parameters["sigma_x"],
                size=self._parameters["n_samples"],
            )
            W = a + b * X + e

        beta0 = np.repeat(self._parameters["beta_0"], Ns)
        logit = self._parameters["beta_x"] * X + beta0
        if self._parameters["beta_z"] is not None:
            Z = rng.normal(
                self._parameters["mu_z"],
                self._parameters["sigma_z"],
                size=(np.sum(Ns), len(self._parameters["beta_z"])),
            )
            zint = (np.array(self._parameters["beta_z"]) * Z).sum(axis=1)
            logit += zint

        if self._typ_outcome == "continue":
            sigma_y = np.repeat(self._parameters["sigma_y"], Ns)
            Y = rng.normal(logit, sigma_y)
        elif self._typ_outcome == "binary":
            p = 1 / (np.exp(-logit) + 1)
            Y = rng.binomial(1, p)

        X_obs = X.copy()
        start = 0
        for ni, nxi in zip(Ns, nKnowX):
            end = start + ni
            if ni < nxi:
                raise ValueError("nsamples(%d) < nKnowX(%d)" % (ni, nxi))
            elif ni > nxi:
                if self._parameters["n_knowX_balance"]:
                    Yi = Y[start:end]
                    y1_ind = np.nonzero(Yi == 1)[0]
                    y0_ind = np.nonzero(Yi == 0)[0]
                    nan_ind1 = rng.choice(
                        y1_ind,
                        int((ni - nxi) / ni * len(y1_ind)),
                        replace=False,
                    )
                    nan_ind0 = rng.choice(
                        y0_ind,
                        int((ni - nxi) / ni * len(y0_ind)),
                        replace=False,
                    )
                    nan_ind = np.concatenate([nan_ind0, nan_ind1]) + start
                else:
                    nan_ind = rng.choice(
                        np.arange(start, end), ni - nxi, replace=False
                    )
                X_obs[nan_ind] = np.NaN
            start = end

        res = pd.DataFrame(
            {
                "W": W,
                "X_true": X,
                "Y": Y,
                "X": X_obs,
                # 从1开始
                "S": np.repeat(
                    np.arange(1, self._parameters["n_studies"] + 1), Ns
                ),
                "H": ~np.isnan(X_obs),
            }
        )
        if self._parameters["beta_z"] is not None:
            res = pd.concat(
                [
                    res,
                    pd.DataFrame(
                        Z, columns=[f"Z{i+1}" for i in range(Z.shape[1])]
                    ),
                ],
                axis=1,
            )
        return res
