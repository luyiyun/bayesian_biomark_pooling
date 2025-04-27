import logging
import os
from typing import Literal, Optional, Sequence, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy import integrate, optimize, special, stats


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


@dataclass
class Simulator:
    """可以为每个studies指定不同的样本量、可观测X样本量等。"""

    mu_x: float = 0
    sigma2_x: float = 1
    OR: float = 1.25
    prevalence: Optional[float] = None
    beta0: Union[float, Sequence[float]] = 1.0
    a: Sequence[float] = (-3, 1, -1, 3)
    b: Sequence[float] = (0.5, 0.75, 1.25, 1.5)
    sigma2_e: Union[float, Sequence[float]] = 1.0
    n_sample_per_studies: Union[int, Sequence[int]] = 1000
    n_knowX_per_studies: Union[int, Sequence[int]] = 100
    n_knowX_balance: bool = False
    direction: Literal["x->w", "w->x"] = "x->w"
    mu_z: float = 0
    sigma2_z: float = 1
    betaz: Sequence[float] | None = None

    # @classmethod
    # def sample_studies
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
    # :
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

    # def __init__(
    #     self,
    #     mu_x: float = 0,
    #     sigma2_x: float = 1,
    #     OR: float = 1.25,
    #     prevalence: Optional[float] = None,
    #     beta0: Union[float, Sequence[float]] = 1.0,
    #     a: Sequence[float] = [-3, 1, -1, 3],
    #     b: Sequence[float] = [0.5, 0.75, 1.25, 1.5],
    #     sigma2_e: Union[float, Sequence[float]] = 1.0,
    #     n_sample_per_studies: Union[int, Sequence[int]] = 1000,
    #     n_knowX_per_studies: Union[int, Sequence[int]] = 100,
    #     n_knowX_balance: bool = False,
    #     direction: Literal["x->w", "w->x"] = "x->w",
    # ) -> None:

    def __post_init__(self):
        logger = logging.getLogger("main.simulate")

        assert self.direction in ["x->w", "w->x"], "direction must be x->w or w->x"
        # 使用OR值来确定beta1
        self.beta1 = np.log(self.OR)
        # 如果指定了prevalence，则使用prevalence来计算beta0，beta0在所有
        # studies中都是一样的
        if self.prevalence is not None:
            # compute the suitable beta0 to produce necessary prevalence
            self.beta0 = get_beta0_by_prevalence(
                self.prevalence, self.beta1, self.mu_x, self.sigma2_x
            )
            logger.info(
                "(pid:%d)Get the beta0 = %.4f by prevalence %.4f"
                % (os.getpid(), self.beta0, self.prevalence)
            )

        # 通过某些参数的数量，来确定studies的数量
        # 同时需要保证这些参数的数量是一致的
        #  (要么是1, 表示在所有studies中一样, 要么是相同的长度)
        params_may_multiple = [
            self.beta0,
            self.a,
            self.b,
            self.sigma2_e,
            self.n_sample_per_studies,
            self.n_knowX_per_studies,
        ]
        ns = None
        for parami in params_may_multiple:
            if isinstance(parami, (int, float)):
                continue
            len_parami = len(parami)
            if ns is None:
                ns = len_parami
            else:
                assert ns == len_parami, (
                    "the length of a, b, sigma2_e, beta0, beta1 must be one or equal."
                )
        assert ns is not None, (
            "the number of studies can not be identified by simulation settings."
        )
        (
            self.beta0,
            self.a,
            self.b,
            self.sigma2_e,
            self.n_sample_per_studies,
            self.n_knowX_per_studies,
        ) = [
            np.array([x] * ns) if isinstance(x, (float, int)) else np.array(x)
            for x in params_may_multiple
        ]

        # self._parameters = {
        #     "mu_x": self.mu_x,
        #     "sigma2_x": self.sigma2_x,
        #     "sigma_x": np.sqrt(self.sigma2_x),
        #     "OR": self.OR,
        #     "beta1": beta1,
        #     "beta0": self.beta0,
        #     "a": self.a,
        #     "b": self.b,
        #     "sigma2_e": self.sigma2_e,
        #     "sigma_e": np.sqrt(self.sigma2_e),
        #     "n_sample_per_studies": self.n_sample_per_studies,
        #     "n_knowX_per_studies": self.n_knowX_per_studies,
        #     "n_studies": ns,
        #     "direction": self.direction,
        #     "n_samples": np.sum(self.n_sample_per_studies),
        #     "n_knowX_balance": self.n_knowX_balance,
        # }

        if self.direction == "w->x":
            self.mu_w = (self.mu_x - self.a) / self.b
            self.sigma2_w = (self.sigma2_x - self.sigma2_e) / (self.b**2)

            # self._parameters.update(
            #     {
            #         "mu_w": mu_w,
            #         "sigma2_w": sigma2_w,
            #         "sigma_w": np.sqrt(sigma2_w),
            #     }
            # )

        self.name = ("prev%.2f-OR%.2f" % (self.prevalence, self.OR)).replace(".", "_")

    # @property
    # def parameters(self):
    #     return self._parameters

    # @property
    # def name(self):
    #     return self.name

    def simulate(self, seed: int):
        rng = np.random.default_rng(seed)

        a = np.repeat(self.a, self.n_sample_per_studies)
        b = np.repeat(self.b, self.n_sample_per_studies)
        sigma_e = np.repeat(self.sigma2_e**0.5, self.n_sample_per_studies)
        e = rng.normal(0, sigma_e)
        if self.direction == "w->x":
            mu_w = np.repeat(self.mu_w, self.n_sample_per_studies)
            sigma_w = np.repeat(self.sigma2_w**0.5, self.n_sample_per_studies)
            W = rng.normal(mu_w, sigma_w)
            X = a + b * W + e
        else:  # x->w
            X = rng.normal(
                self.mu_x, self.sigma2_x**0.5, size=sum(self.n_sample_per_studies)
            )
            W = a + b * X + e

        if self.betaz is not None:
            Z = rng.normal(
                self.mu_z,
                self.sigma2_z**0.5,
                size=(sum(self.n_sample_per_studies), len(self.betaz)),
            )

        beta0 = np.repeat(self.beta0, self.n_sample_per_studies)
        logit = self.beta1 * X + beta0
        if self.betaz is not None:
            logit += np.dot(Z, self.betaz)
        p = 1 / (np.exp(-logit) + 1)
        Y = rng.binomial(1, p)

        X_obs = X.copy()
        start = 0
        for ni, nxi in zip(self.n_sample_per_studies, self.n_knowX_per_studies):
            end = start + ni
            if ni < nxi:
                raise ValueError("nsamples(%d) < nKnowX(%d)" % (ni, nxi))
            elif ni > nxi:
                if self.n_knowX_balance:
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
                    nan_ind = rng.choice(np.arange(start, end), ni - nxi, replace=False)
                X_obs[nan_ind] = np.nan
            start = end

        res = pd.DataFrame(
            {
                "W": W,
                "X_true": X,
                "Y": Y,
                "X": X_obs,
                # 从1开始
                "S": np.repeat(
                    np.arange(1, len(self.n_sample_per_studies) + 1),
                    self.n_sample_per_studies,
                ),
                "H": ~np.isnan(X_obs),
            }
        )
        if self.betaz is not None:
            for i in range(len(self.betaz)):
                res[f"Z{i + 1}"] = Z[:, i]
        return res
