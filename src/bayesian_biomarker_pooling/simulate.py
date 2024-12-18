import logging

import json
from collections import abc
from dataclasses import dataclass, asdict
from typing import Literal

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


@dataclass
class Simulator:
    """可以为每个studies指定不同的样本量、可观测X样本量等。"""

    n_studies: int = 4
    n_samples: int | tuple[int, ...] = 100
    ratio_observed_X: float | tuple[float, ...] = 0.1
    balance_observed_X: bool = False
    outcome_type: Literal["binary", "continue"] = "binary"
    direction: Literal["x->w", "w->x"] = "x->w"
    mu_x: float = 0.0
    sigma2_x: float = 1.0
    beta_x: float = 1.0
    beta_0: float | tuple[float, ...] = 1.0
    a: tuple[float, ...] = (-3, 1, -1, 3)
    b: tuple[float, ...] = (0.5, 0.75, 1.25, 1.5)
    sigma2_e: float | tuple[float, ...] = 1.0
    sigma2_y: float | tuple[float, ...] = 1.0
    prevalence: float | tuple[float, ...] | None = None
    n_z: int = 0
    beta_z: float | tuple[int, ...] | None = None
    mu_z: float | tuple[float, ...] | None = 0.0
    sigma_z: float | tuple[float, ...] | None = 1.0

    def __post_init__(self):
        # ============== 输入参数检查 ==============
        assert self.direction in [
            "x->w",
            "w->x",
        ], "direction must be x->w or w->x"
        assert self.outcome_type in [
            "binary",
            "continue",
        ], "outcome_type must be binary or continue"

        if self.outcome_type == "continue":
            assert (
                self.beta_0 is not None
            ), "beta_0 must be specified for continue outcome."
        elif self.outcome_type == "binary":
            assert (self.prevalence is None and self.beta_0 is not None) or (
                self.prevalence is not None and self.beta_0 is None
            ), "can not set prevalence and beta0 simultaneously."

        for param_name in [
            "n_samples",
            "ratio_observed_X",
            "beta_0",
            "a",
            "b",
            "sigma2_e",
            "sigma2_y",
        ]:
            param = getattr(self, param_name)
            if isinstance(param, abc.Sequence):
                assert len(param) == self.n_studies, (
                    f"the length of {param_name} must "
                    "be equal to n_studies."
                )

        if self.n_z > 0:
            if isinstance(self.beta_z, abc.Sequence):
                assert (
                    len(self.beta_z) == self.n_z
                ), "the length of beta_z must be equal to n_z."
            if isinstance(self.mu_z, abc.Sequence):
                assert (
                    len(self.mu_z) == self.n_z
                ), "the length of mu_z must be equal to n_z."
            if isinstance(self.sigma_z, abc.Sequence):
                assert (
                    len(self.sigma_z) == self.n_z
                ), "the length of sigma_z must be equal to n_z."

        # ============== 由prevalence计算beta0 ==============
        # 如果指定了prevalence，则使用prevalence来计算beta0，prevalence也可以指定多个，来对应多个beta0
        if self.outcome_type == "binary" and self.prevalence is not None:
            if isinstance(self.prevalence, (int, float)):
                beta_0 = get_beta0_by_prevalence(
                    self.prevalence, self.beta_x, self.mu_x, self.sigma2_x
                )
                logger.info(
                    f"Get the beta0 = {beta_0:.4f} "
                    f"by prevalence {self.prevalence:.4f}"
                )
            elif isinstance(self.prevalence, abc.Sequence):
                beta_0 = [
                    get_beta0_by_prevalence(
                        pi, self.beta_x, self.mu_x, self.sigma2_x
                    )
                    for pi in self.prevalence
                ]
                logger.info(
                    f"Get the beta0 = {str(beta_0)} "
                    f"by prevalence {self.prevalence}"
                )
            else:
                raise ValueError("prevalence must be a scalar or a sequence.")
            self.beta_0 = beta_0

        # ============== 参数长度修正 ==============
        for param_name in [
            "n_samples",
            "ratio_observed_X",
            "beta_0",
            "a",
            "b",
            "sigma2_e",
            "sigma2_y",
            "prevalence",
        ]:
            param = getattr(self, param_name)
            if isinstance(param, (float, int)):
                setattr(self, param_name, [param] * self.n_studies)

        for param_name in ["beta_z", "mu_z", "sigma_z"]:
            param = getattr(self, param_name)
            if isinstance(param, (float, int)):
                setattr(self, param_name, [param] * self.n_z)

        # ============== 相关参数转换为ndarray ==============
        for param_name in [
            "n_samples",
            "ratio_observed_X",
            "beta_0",
            "a",
            "b",
            "sigma2_e",
            "sigma2_y",
            "prevalence",
            "beta_z",
            "mu_z",
            "sigma_z",
        ]:
            param = getattr(self, param_name)
            if isinstance(param, abc.Sequence):
                setattr(self, param_name, np.array(param))

        # ============== 参数相关转换 ==============
        self.sigma_y = np.sqrt(self.sigma2_y)
        self.sigma_e = np.sqrt(self.sigma2_e)
        self.sigma_x = np.sqrt(self.sigma2_x)
        self.n_samples_total = np.sum(self.n_samples)

        # ============== w->x相关参数计算 ==============
        if self.direction == "w->x":
            self.mu_w = (self.mu_x - self.a) / self.b
            self.sigma2_w = (self.sigma2_x - self.sigma2_e) / (self.b**2)
            self.sigma_w = np.sqrt(self.sigma2_w)

    def save(self, fn: str) -> None:
        if not fn.endswith(".json"):
            raise ValueError("just support json format.")

        params = asdict(self)
        params = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in params.items()
        }
        with open(fn, "w") as f:
            json.dump(params, f, indent=4)

    @classmethod
    def load(cls, fn: str) -> "Simulator":
        if not fn.endswith(".json"):
            raise ValueError("just support json format.")

        with open(fn, "r") as f:
            params = json.load(f)
        return cls(**params)

    @property
    def parameters(self):
        keys, values = [], []
        for k in [
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
            if not hasattr(self, k):
                continue

            param = getattr(self, k)
            if isinstance(param, (float, int)) or (
                isinstance(param, np.ndarray) and len(param) == 1
            ):
                keys.append(k)
                values.append(param)
            elif isinstance(param, np.ndarray):
                keys.extend([f"{k}-{i}" for i in range(len(param))])
                values.append(param)

        return pd.Series(np.concatenate(values), index=keys)

    def simulate(self, seed: int | None = None):
        rng = np.random.default_rng(seed)

        nKnowX = (self.n_samples * self.ratio_observed_X).astype(int)

        a = np.repeat(self.a, self.n_samples)  # 111 222 333 444
        b = np.repeat(self.b, self.n_samples)
        sigma_e = np.repeat(self.sigma_e, self.n_samples)
        e = rng.normal(0, sigma_e)
        if self.direction == "w->x":
            mu_w = np.repeat(self.mu_w, self.n_samples)
            sigma_w = np.repeat(self.sigma_w, self.n_samples)
            W = rng.normal(mu_w, sigma_w)
            X = a + b * W + e
        else:  # x->w
            X = rng.normal(
                self.mu_x,
                self.sigma_x,
                size=self.n_samples_total,
            )
            W = a + b * X + e

        beta0 = np.repeat(self.beta_0, self.n_samples)
        logit = self.beta_x * X + beta0
        if self.n_z > 0:
            Z = rng.normal(
                self.mu_z,
                self.sigma_z,
                size=(self.n_samples_total, len(self.beta_z)),
            )
            zint = (np.array(self.beta_z) * Z).sum(axis=1)
            logit += zint

        if self.outcome_type == "continue":
            sigma_y = np.repeat(self.sigma_y, self.n_samples_total)
            Y = rng.normal(logit, sigma_y)
        elif self.outcome_type == "binary":
            p = 1 / (np.exp(-logit) + 1)
            Y = rng.binomial(1, p)

        X_obs = X.copy()
        start = 0
        for ni, nxi in zip(self.n_samples, nKnowX):
            end = start + ni
            if ni < nxi:
                raise ValueError("nsamples(%d) < nKnowX(%d)" % (ni, nxi))
            elif ni > nxi:
                if self.balance_observed_X:
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
                    np.arange(1, self.n_studies + 1), self.n_samples
                ),
                "H": ~np.isnan(X_obs),
            }
        )
        if self.n_z > 0:
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


if __name__ == "__main__":
    simulator = Simulator()
    d = asdict(simulator)
    print(d)
    simulator.save("test.json")
    new_simulator = Simulator.load("test.json")
    print(asdict(new_simulator))

    dat = new_simulator.simulate(seed=1)
    print(dat.shape)
    print(dat.head())
