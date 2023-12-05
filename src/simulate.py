import logging
import os
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
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


def simulate(
    seed: Optional[int] = 0,
    Ns: int = 1000,
    n_knowX: int = 100,
    mu_x: float = 0,
    sigma2_x: float = 1,
    prevalence: Optional[float] = None,
    a: Sequence[float] = [-3, 1, -1, 3],
    b: Sequence[float] = [0.5, 0.75, 1.25, 1.5],
    sigma2_e: Union[float, Sequence[float]] = 1.0,
    beta0: Union[float, Sequence[float]] = 1.0,
    beta1: float = np.log(1.25),
    direction: Literal["x->w", "w->x"] = "x->w",
) -> Tuple[pd.DataFrame, Dict]:
    assert direction in ["x->w", "w->x"]
    logger = logging.getLogger("main.simulate")

    ns = None
    for parami in [a, b, sigma2_e, beta0, beta1]:
        if isinstance(parami, (int, float)):
            continue
        len_parami = len(parami)
        if ns is None:
            ns = len_parami
            continue
        else:
            assert ns == len_parami, (
                "the length of a, b, sigma2_e, "
                "beta0, beta1 must be one or equal."
            )

    if prevalence is not None:
        # compute the suitable beta0 to produce necessary prevalence
        beta0 = get_beta0_by_prevalence(prevalence, beta1, mu_x, sigma2_x)
        logger.info(
            "(pid:%d)Get the beta0 = %.4f by prevalence %.4f"
            % (os.getpid(), beta0, prevalence)
        )

    a, b, sigma2_e, beta0, beta1 = (
        np.array(x) for x in [a, b, sigma2_e, beta0, beta1]
    )
    rng = np.random.default_rng(seed)

    if direction == "w->x":
        mu_w = (mu_x - a) / b
        sigma2_w = (sigma2_x - sigma2_e) / (b**2)

        W = rng.normal(mu_w, np.sqrt(sigma2_w), size=(Ns, ns))
        e = rng.normal(0, np.sqrt(sigma2_e), size=(Ns, ns))
        X = a + b * W + e
    else:  # x->w
        X = rng.normal(mu_x, np.sqrt(sigma2_x), size=(Ns, ns))
        e = rng.normal(0, np.sqrt(sigma2_e), size=(Ns, ns))
        W = a + b * X + e

    logit = beta1 * X + beta0
    p = 1 / (np.exp(-logit) + 1)
    Y = rng.binomial(1, p)

    X_obs = X.copy()
    for i in range(ns):
        nan_ind = rng.choice(Ns, Ns - n_knowX, replace=False)
        X_obs[nan_ind, i] = np.NaN

    res = pd.DataFrame(
        {
            "W": W.flatten(),
            "X_true": X.flatten(),
            "Y": Y.flatten(),
            "X": X_obs.flatten(),
            "S": np.tile(np.arange(ns), Ns) + 1,  # 从1开始
            "H": ~np.isnan(X_obs.flatten())
        }
    )

    return res, {
        "betax": beta1,
        "a_s": a,
        "b_s": b,
    }
