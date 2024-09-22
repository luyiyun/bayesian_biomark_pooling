from typing import Literal, Optional, Tuple, Union, List, Sequence

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pymc.backends.base import MultiTrace

from .base import BiomarkerPoolBase, check_data


"""
Z --> Y <-- X --> W
TODO:
2. Z -?-> W;
3. binary, continue, and survival outcomes;
4. +multiple imputation.
"""


class BBPResults:

    all_hidden_variables: List[str] = [
        "betax",
        "mu_sigma_w",
        "sigma_sigma_w",
        "simga_ws",
        "a",
        "b",
        "beta0",
        "sigma_a",
        "sigma_b",
        "sigma_0",
        "a_s",
        "b_s",
        "beta0s",
        "mu_x",
        "sigma_x",
        "X_no_obs",
        "betaz",
    ]

    def __init__(
        self,
        model: pm.Model,
        res_mcmc: Union[az.InferenceData, MultiTrace, pm.Approximation],
    ) -> None:
        self.model_ = model
        self.res_mcmc_ = res_mcmc

    def summary(
        self,
        var_names: Optional[Tuple[str]] = ("betax",),
    ) -> pd.DataFrame:

        str_all_hidden_variables = ",".join(self.all_hidden_variables)
        assert (
            len(set(var_names).difference(self.all_hidden_variables)) == 0
        ), (
            "The element of var_names must "
            f"be one of [{str_all_hidden_variables}]"
        )

        res_df = az.summary(
            self.res_mcmc_,
            hdi_prob=0.95,
            kind="stats",
            var_names=list(var_names),
        )

        return res_df


class BBP(BiomarkerPoolBase):

    def __init__(
        self,
        prior_betax_std: Union[Literal["inf"], float] = "inf",
        prior_betaz_std: Union[Literal["inf"], float] = "inf",
        prior_sigma_ws: Literal["gamma", "inv_gamma"] = "gamma",
        prior_sigma_ab0: Literal["half_cauchy", "half_flat"] = "half_cauchy",
        std_prior_a: float = 1.0,
        std_prior_b: float = 1.0,
        std_prior_beta0: float = 1.0,
        std_prior_mu_x: float = 10.0,
        mean_prior_sigma_x: float = 1.0,
        sigma_prior_sigma_x: float = 1.0,
        solver: Literal["pymc", "vi", "blackjax"] = "pymc",
        nsample: int = 1000,
        ntunes: int = 1000,
        nchains: int = 4,
        pbar: bool = True,
        seed: int = 0,
        target_accept: Optional[float] = None,
    ) -> None:

        assert prior_betax_std == "inf" or (
            isinstance(prior_betax_std, float) and prior_betax_std > 0
        )
        assert prior_betaz_std == "inf" or (
            isinstance(prior_betaz_std, float) and prior_betaz_std > 0
        )
        assert prior_sigma_ws in ["gamma", "inv_gamma"]
        assert prior_sigma_ab0 in ["half_cauchy", "half_flat"]
        assert solver in ["pymc", "vi", "blackjax"]

        self.prior_betax_std_ = prior_betax_std
        self.prior_betaz_std_ = prior_betaz_std
        self.prior_sigma_ws_ = prior_sigma_ws
        self.prior_sigma_ab0_ = prior_sigma_ab0
        self.std_prior_a_ = std_prior_a
        self.std_prior_b_ = std_prior_b
        self.std_prior_beta0_ = std_prior_beta0
        self.std_prior_mu_x_ = std_prior_mu_x
        self.mean_prior_sigma_x_ = mean_prior_sigma_x
        self.sigma_prior_sigma_x_ = sigma_prior_sigma_x
        self.solver_ = solver
        self.nsample_ = nsample
        self.ntunes_ = ntunes
        self.nchains_ = nchains
        self.pbar_ = pbar
        self.seed_ = seed
        self.target_accept_ = target_accept

    def fit(
        self,
        df: pd.DataFrame,
        X_col: str = "X",
        S_col: str = "S",
        W_col: str = "W",
        Y_col: str = "Y",
        Z_col: Optional[Union[str, Sequence[str]]] = None,
    ) -> BBPResults:

        X, S, W, Y, Z = check_data(df, X_col, S_col, W_col, Y_col, Z_col)

        ##################################################################
        # split the data
        ##################################################################
        unique_S = np.unique(S)
        is_m = pd.isnull(X)
        is_o = np.logical_not(is_m)

        # n_S = len(unique_S)
        # n_xKnow = int(is_o.sum())
        n_xUnKnow = int(is_m.sum())

        X_Know = X[is_o]

        XWYZ_xKnow, WYZ_xUnKnow = {}, {}
        for si in unique_S:
            is_o_si = (S == si) & is_o
            if is_o_si.sum() == 0:
                XWYZ_xKnow[si] = None
            else:
                item = {"X": X[is_o_si], "W": W[is_o_si], "Y": Y[is_o_si]}
                if Z is not None:
                    item["Z"] = Z[is_o_si, :]
                XWYZ_xKnow[si] = item

            is_m_si = (S == si) & is_m
            if is_m_si.sum() == 0:
                WYZ_xUnKnow[si] = None
            else:
                item = {"X": X[is_m_si], "W": W[is_m_si], "Y": Y[is_m_si]}
                if Z is not None:
                    item["Z"] = Z[is_m_si, :]
                WYZ_xUnKnow[si] = item

        ##################################################################
        # create pymc model
        ##################################################################
        coords = {"S": unique_S}
        if Z is not None:
            coords["Z"] = [Z_col] if isinstance(Z_col, str) else Z_col

        with pm.Model(coords=coords) as model:
            # ============= 1. set prior =============
            # 1.1 betax
            betax = (
                pm.Flat("betax")
                if self.prior_betax_std_ == "inf"
                else pm.Normal("betax", mu=0, sigma=self.prior_betax_std_)
            )
            # 1.2 sigma_w_s
            mu_sigma_w = pm.HalfFlat("mu_sigma_w")  # mu_sigma_w这里其实是mode
            sigma_sigma_w = pm.HalfCauchy("sigma_sigma_w", 1.0)
            if self.prior_sigma_ws_ == "gamma":
                sigma_ws = pm.Gamma(
                    "sigma_ws", mu=mu_sigma_w, sigma=sigma_sigma_w, dims="S"
                )
            elif self.prior_sigma_ws_ == "inv_gamma":
                sigma2_ws = pm.InverseGamma(
                    "sigma2_ws", mu=mu_sigma_w, sigma=sigma_sigma_w, dims="S"
                )
                sigma_ws = pm.Deterministic(
                    "sigma_ws", pm.math.sqrt(sigma2_ws)
                )
            # 1.3 a_s, b_s, beta0_s
            a = pm.Normal("a", 0, self.std_prior_a_)
            b = pm.Normal("b", 0, self.std_prior_b_)
            beta0 = pm.Normal("beta0", 0, self.std_prior_beta0_)
            if self.prior_sigma_ab0_ == "half_cauchy":
                sigma_a = pm.HalfCauchy("sigma_a", 1.0)
                sigma_b = pm.HalfCauchy("sigma_b", 1.0)
                sigma_0 = pm.HalfCauchy("sigma_0", 1.0)
            elif self.prior_sigma_ab0_ == "half_flat":
                sigma_a = pm.HalfFlat("sigma_a")
                sigma_b = pm.HalfFlat("sigma_b")
                sigma_0 = pm.HalfFlat("sigma_0")
            a_s = pm.Normal("a_s", a, sigma_a, dims="S")
            b_s = pm.Normal("b_s", b, sigma_b, dims="S")
            beta0s = pm.Normal("beta0s", beta0, sigma_0, dims="S")
            # 1.4 mu_x, sigma_x
            mu_x = pm.Normal("mu_x", 0, self.std_prior_mu_x_)
            # convert the mean-sigma format to alpha-beta format
            beta = (
                self.mean_prior_sigma_x_
                + np.sqrt(
                    self.mean_prior_sigma_x_ + self.sigma_prior_sigma_x_**2
                )
            ) / (2 * self.sigma_prior_sigma_x_**2)
            alpha = self.mean_prior_sigma_x_ * beta + 1
            sigma_x = pm.Gamma("sigma_x", alpha=alpha, beta=beta)
            # 1.5 beta_z
            if Z is not None:
                beta_z = (
                    pm.Flat("betaz")
                    if self.prior_betaz_std_ == "inf"
                    else pm.Normal(
                        "betaz", mu=0, sigma=self.prior_betaz_std_, dims="Z"
                    )
                )

            # if beta0s.ndim == 0:
            #     beta0s = [beta0s] * n_S
            # if sigma_ws.ndim == 0:
            #     sigma_ws = [sigma_ws] * n_S

            # ============= 2. set data generation model =============
            # for samples that can not see X
            X_no_obs = pm.Normal(
                "X_no_obs",
                mu_x,
                sigma_x,
                size=n_xUnKnow,
            )
            start = 0
            for i, (si, data_dict) in enumerate(WYZ_xUnKnow.items()):
                if data_dict is None:
                    continue
                Wi, Yi = data_dict["W"], data_dict["Y"]
                end = start + Wi.shape[0]
                X_no_obs_i = X_no_obs[start:end]
                mu_W_i = a_s[i] + b_s[i] * X_no_obs_i  # 只接受iloc索引
                # TODO: Z是否会影响Y?
                # if Z is not None:
                #     mu_Y_i += beta_z * Z
                pm.Normal(
                    f"W_{si}_no_obs_X",
                    mu_W_i,
                    sigma_ws[i],
                    observed=Wi,
                )
                zpred = (
                    0.0 if Z is None else (beta_z * data_dict["Z"]).sum(axis=1)
                )
                pred = beta0s[i] + betax * X_no_obs_i + zpred
                pm.Bernoulli(
                    f"Y_{si}_no_obs_X",
                    logit_p=pred,
                    observed=Yi,
                )
                start = end

            # for samples that can see X
            pm.Normal("X_obs", mu_x, sigma_x, observed=X_Know)
            for i, (si, data_dict) in enumerate(XWYZ_xKnow.items()):
                if data_dict is None:
                    continue
                Xi, Wi, Yi = data_dict["X"], data_dict["W"], data_dict["Y"]
                pm.Normal(
                    f"W_{si}_obs_X",
                    a_s[i] + b_s[i] * Xi,
                    sigma_ws[i],
                    observed=Wi,
                )
                zpred = (
                    0 if Z is None else (beta_z * data_dict["Z"]).sum(axis=1)
                )
                pred = beta0s[i] + betax * Xi + zpred
                pm.Bernoulli(
                    f"Y_{si}_obs_X",
                    logit_p=pred,
                    observed=Yi,
                )

            # ============= 3. MCMC sampling =============
            if self.solver_ == "vi":
                res = pm.fit(progressbar=self.pbar_, random_seed=self.seed_)
            else:
                kwargs = dict(
                    tune=self.ntunes_,
                    chains=self.nchains_,
                    cores=self.nchains_,
                    progressbar=self.pbar_,
                    random_seed=list(
                        range(self.seed_, self.seed_ + self.nchains_)
                    ),
                    nuts_sampler=self.solver_,
                )
                if self.target_accept_ is not None:
                    kwargs["target_accept"] = self.target_accept_
                res = pm.sample(self.nsample_, **kwargs)

        return BBPResults(model, res)
