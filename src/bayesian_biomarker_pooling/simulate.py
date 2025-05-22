import logging
import os
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import integrate, optimize, special, stats


logger = logging.getLogger(__name__)


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


def default_serializer(obj):
    """Custom serializer for objects not serializable by default json encoder."""
    if isinstance(obj, np.ndarray):
        # Convert ndarray to a list and then to a tuple to ensure hashability
        # and consistent ordering for the purpose of generating a unique name.
        # Using a tuple of its shape and a tuple of its flattened elements
        # ensures that both shape and content are considered.
        return (obj.shape, tuple(obj.flatten().tolist()))
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


@dataclass
class Simulator:
    """the base class for simulating data from a bayesian biomarker pooling model"""

    mu_x: float = 0
    sigma2_x: float = 1
    beta0: Union[float, Sequence[float]] = 1.0
    a: Sequence[float] = (-3, 1, -1, 3)
    b: Sequence[float] = (0.5, 0.75, 1.25, 1.5)
    sigma2_e: Union[float, Sequence[float]] = 1.0
    n_sample_per_studies: Union[int, Sequence[int]] = 1000
    n_knowX_per_studies: Union[int, Sequence[int]] = 100
    direction: Literal["x->w", "w->x"] = "x->w"
    mu_z: float = 0
    sigma2_z: float = 1
    betaz: Sequence[float] | None = None

    def __post_init__(self):
        # logger = logging.getLogger("main.simulate")

        assert self.direction in ["x->w", "w->x"], "direction must be x->w or w->x"

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

        if self.direction == "w->x":
            self.mu_w = (self.mu_x - self.a) / self.b
            self.sigma2_w = (self.sigma2_x - self.sigma2_e) / (self.b**2)

    @property
    def name(self) -> str:
        serialized_dict = json.dumps(
            asdict(self), sort_keys=True, default=default_serializer
        )
        # Create a SHA-256 hash object
        hasher = hashlib.sha256()
        # Update the hasher with the serialized dictionary string (encoded to bytes)
        hasher.update(serialized_dict.encode("utf-8"))
        # Get the hexadecimal representation of the hash
        return hasher.hexdigest()

    def generate_Y(self, rng, X, W, Y_, Z) -> np.ndarray:
        raise NotImplementedError

    def generate_mask(self, rng, X, W, Y, Z) -> np.ndarray:
        raise NotImplementedError

    def simulate(self, seed: int | None = None) -> pd.DataFrame:
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
        else:
            Z = None

        beta0 = np.repeat(self.beta0, self.n_sample_per_studies)
        Y_ = self.beta1 * X + beta0

        if self.betaz is not None:
            Y_ += np.dot(Z, self.betaz)

        Y = self.generate_Y(rng, X, W, Y_, Z)
        mask = self.generate_mask(rng, X, W, Y, Z)

        X_obs = X.copy()
        X_obs[mask] = np.nan

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
                "H": ~mask,
            }
        )
        if self.betaz is not None:
            for i in range(len(self.betaz)):
                res[f"Z{i + 1}"] = Z[:, i]
        return res


@dataclass
class BinarySimulator(Simulator):
    """the simulator for binary outcome"""

    OR: float = 1.25
    prevalence: Optional[float] = None
    n_knowX_balance: bool = False

    def __post_init__(self):
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

        # post_init中需要更新后的beta0
        super().__post_init__()

    def generate_Y(self, rng, X, W, Y_, Z) -> np.ndarray:
        p = 1 / (np.exp(-Y_) + 1)
        Y = rng.binomial(1, p)
        return Y

    def generate_mask(self, rng, X, W, Y, Z) -> np.ndarray:
        mask = np.zeros_like(Y, dtype=bool)
        start = 0
        for ni, nxi in zip(self.n_sample_per_studies, self.n_knowX_per_studies):
            end = start + ni
            if ni < nxi:
                raise ValueError("nsamples(%d) < nKnowX(%d)" % (ni, nxi))
            elif ni > nxi:
                if self.n_knowX_balance:
                    Yi = Y[start:end]
                    n_nan = ni - nxi
                    n_nan_0 = int(n_nan * (Yi == 0).mean())
                    n_nan_1 = n_nan - n_nan_0
                    nan_ind0 = rng.choice(
                        np.nonzero(Yi == 0)[0], n_nan_0, replace=False
                    )
                    nan_ind1 = rng.choice(
                        np.nonzero(Yi == 1)[0], n_nan_1, replace=False
                    )
                    nan_ind = np.concatenate([nan_ind0, nan_ind1]) + start
                else:
                    nan_ind = rng.choice(np.arange(start, end), ni - nxi, replace=False)
                mask[nan_ind] = True
            start = end
        return mask


@dataclass
class ContinuousSimulator(Simulator):
    """the simulator for continuous outcome"""

    beta1: float = 1.0
    sigma2_y: float = 1.0

    def generate_Y(self, rng, X, W, Y_, Z):
        sigma_y = np.repeat(self.sigma2_y**0.5, Y_.shape[0])
        y = rng.normal(Y_, sigma_y)
        return y

    def generate_mask(self, rng, X, W, Y, Z):
        mask = np.zeros_like(Y, dtype=bool)
        start = 0
        for ni, nxi in zip(self.n_sample_per_studies, self.n_knowX_per_studies):
            end = start + ni
            if ni < nxi:
                raise ValueError("nsamples(%d) < nKnowX(%d)" % (ni, nxi))
            elif ni > nxi:
                nan_ind = rng.choice(np.arange(start, end), ni - nxi, replace=False)
                mask[nan_ind] = True
            start = end
        return mask


if __name__ == "__main__":
    simulator1 = BinarySimulator()
    df1 = simulator1.simulate(seed=1)
    print(simulator1.name)
    print(df1.head())

    simulator2 = ContinuousSimulator()
    df2 = simulator2.simulate(seed=1)
    print(simulator2.name)
    print(df2.head())
