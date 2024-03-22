import logging
import os
import hashlib
from collections import abc

from typing import Literal, Optional, Sequence, Union, Tuple, Dict

import numpy as np
import pandas as pd

# import scipy.stats as ss
from scipy import integrate, optimize, special, stats


def prepare_params_source_heterogeneity(
    *params: Union[float, Sequence[float]]
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

    # @classmethod
    # def sample_studies(
    #     cls,
    #     mu_x: float = 0,
    #     sigma2_x: float = 1,
    #     OR: float = 1.25,
    #     prevalence: Optional[float] = None,
    #     n_studies: int = 4,
    #     a_mu: float = 0.0,
    #     a_sigma: float = 3.0,
    #     b_mu: float = 0.0,
    #     b_sigma: float = 3.0,
    #     sigma2e_shape: float = 1.5,
    #     n_sample_per_studies: Union[int, Sequence[int]] = 1000,
    #     n_knowX_per_studies: Union[int, Sequence[int]] = 100,
    #     n_knowX_balance: bool = False,
    #     direction: Literal["x->w", "w->x"] = "x->w",
    #     seed: int = 0,
    # ):
    #     study_seed = np.random.default_rng(seed)
    #     a = study_seed.normal(a_mu, a_sigma, size=n_studies)
    #     b = study_seed.normal(b_mu, b_sigma, size=n_studies)
    #     sigma2e = ss.invgamma.rvs(
    #         sigma2e_shape, size=n_studies, random_state=study_seed
    #     )
    #     return cls(
    #         mu_x=mu_x,
    #         sigma2_x=sigma2_x,
    #         OR=OR,
    #         prevalence=prevalence,
    #         a=a,
    #         b=b,
    #         sigma2_e=sigma2e,
    #         n_sample_per_studies=n_sample_per_studies,
    #         n_knowX_per_studies=n_knowX_per_studies,
    #         n_knowX_balance=n_knowX_balance,
    #         direction=direction,
    #     )

    def __init__(
        self,
        mu_x: float = 0.0,
        sigma2_x: float = 1.0,
        beta_x: float = 1.0,
        beta_z: Optional[Union[float, Sequence[float]]] = None,
        mu_z: Union[float, Sequence[float]] = 0.0,
        sigma_z: Union[float, Sequence[float]] = 1.0,
        prevalence: Optional[float] = None,
        beta_0: Union[float, Sequence[float]] = 1.0,
        a: Sequence[float] = [-3, 1, -1, 3],
        b: Sequence[float] = [0.5, 0.75, 1.25, 1.5],
        sigma2_e: Union[float, Sequence[float]] = 1.0,
        sigma2_y: Union[float, Sequence[float]] = 1.0,
        n_sample_per_studies: Union[int, Sequence[int]] = 100,
        n_knowX_per_studies: Union[int, Sequence[int]] = 10,
        n_knowX_balance: bool = False,
        direction: Literal["x->w", "w->x"] = "x->w",
        type_outcome: Literal["binary", "continue"] = "binary",
    ) -> None:
        assert direction in ["x->w", "w->x"]
        assert type_outcome in ["binary", "continue"]
        if type_outcome == "continue":
            assert beta_0 is not None
        if beta_z is not None and isinstance(mu_z, abc.Sequence):
            assert len(beta_z) == len(mu_z)
        if beta_z is not None and isinstance(sigma_z, abc.Sequence):
            assert len(beta_z) == len(mu_z)

        logger = logging.getLogger("main.simulate")

        # 如果指定了prevalence，则使用prevalence来计算beta0，beta0在所有
        # studies中都是一样的
        if type_outcome == "binary" and prevalence is not None:
            # compute the suitable beta0 to produce necessary prevalence
            beta_0 = get_beta0_by_prevalence(
                prevalence, beta_x, mu_x, sigma2_x
            )
            logger.info(
                f"(pid:{os.getpid()})Get the beta0 = {beta_0:.4f} "
                f"by prevalence {prevalence:.4f}"
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
        index, res = [], []
        for key in [
            "mu_x",
            "sigma2_x",
            "a",
            "b",
            "sigma2_e",
            "beta_x",
            "beta_0",
            "beta_z",
            "sigma2_y",
        ]:
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
                    n_nan = ni - nxi
                    n_nan_0 = n_nan // 2
                    n_nan_1 = n_nan - n_nan_0
                    Yi = Y[start:end]
                    nan_ind0 = rng.choice(
                        np.nonzero(Yi == 0)[0], n_nan_0, replace=False
                    )
                    nan_ind1 = rng.choice(
                        np.nonzero(Yi == 1)[0], n_nan_1, replace=False
                    )
                    nan_ind = np.concatenate([nan_ind0, nan_ind1])
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
