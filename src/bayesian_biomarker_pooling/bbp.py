from typing import Literal, Optional, Tuple, Union, Sequence

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pymc.backends.base import MultiTrace
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, LogisticRegression
from lifelines import CoxPHFitter

from .base import BiomarkerPoolBase, check_data


"""
Z --> Y <-- X --> W
TODO:
2. Z -?-> W;
4. +multiple imputation.
"""


class BBPResults:

    def __init__(
        self,
        model: pm.Model,
        res_mcmc: Union[az.InferenceData, MultiTrace, pm.Approximation],
        res_multi_imp: Optional[np.ndarray] = None,
    ) -> None:
        self.model_ = model
        self.res_mcmc_ = res_mcmc
        self.res_multi_imp_ = res_multi_imp

        self._all_hidden_vars = list(self.res_mcmc_.posterior.keys())

    def summary(
        self,
        var_names: Optional[Tuple[str]] = ("betax",),
    ) -> pd.DataFrame:

        if len(set(var_names).difference(self._all_hidden_vars)) > 0:
            str_all_hidden_variables = ",".join(self._all_hidden_vars)
            raise ValueError(
                "The element of var_names "
                f"must be one of [{str_all_hidden_variables}]"
            )

        res_df = az.summary(
            self.res_mcmc_,
            hdi_prob=0.95,
            kind="stats",
            var_names=list(var_names),
        )
        if self.res_multi_imp_ is not None:
            if var_names is None or "betax" in var_names:
                val = np.concatenate(
                    [
                        self.res_multi_imp_[:, 0],
                        self.res_mcmc_.posterior["betax"].values.flatten(),
                    ]
                )
                res_df.loc["betax", "mean"] = np.mean(val)
                res_df.loc["betax", "sd"] = np.std(val, ddof=1)
                res_df.loc["betax", ["hdi_2.5%", "hdi_97.5%"]] = np.quantile(
                    val, [0.025, 0.975]
                )

                if self.res_multi_imp_.shape[1] > 1 and (
                    var_names is None or "betaz" in var_names
                ):
                    val_betaz_mcmc = self.res_mcmc_.posterior["betaz"].values
                    val_betaz = np.concatenate(
                        [
                            self.res_multi_imp_[:, 1:],
                            val_betaz_mcmc.reshape(
                                -1, val_betaz_mcmc.shape[-1]
                            ),
                        ],
                        axis=0,
                    )
                    ind = res_df.index.str.startswith("betaz")
                    res_df.loc[ind, "mean"] = np.mean(val_betaz, axis=0)
                    res_df.loc[ind, "sd"] = np.std(val_betaz, axis=0, ddof=1)
                    res_df.loc[ind, ["hdi_2.5%", "hdi_97.5%"]] = np.quantile(
                        val_betaz, [0.025, 0.975], axis=0
                    )

        return res_df


TYPE_STD = Union[Literal["inf"], float]


def set_real_value_rv(
    name: str, std: TYPE_STD, dims: Optional[str] = None
) -> pm.Continuous:
    return (
        pm.Flat(name, dims=dims)
        if std == "inf"
        else pm.Normal(name, mu=0, sigma=std, dims=dims)
    )


def set_pos_value_rv(
    name: str, scale: TYPE_STD, dims: Optional[str] = None
) -> pm.Continuous:
    return (
        pm.HalfFlat(name, dims=dims)
        if scale == "inf"
        else pm.HalfCauchy(name, beta=scale, dims=dims)
    )


def set_surv_likelihood(
    name: str,
    Yi: np.ndarray,
    pred: pm.Continuous,
    shape: Optional[pm.Continuous] = None,
    kind: Literal["expon", "weibull", "piece-const"] = "expon",
    interval_bounds: Optional[np.ndarray] = None,
    lambda0: Optional[pm.Continuous] = None,
) -> pm.Continuous:
    if kind == "expon":
        return pm.Poisson(
            name,
            pm.math.exp(pred) * Yi[:, 0],
            observed=Yi[:, 1].astype(int),
        )
    elif kind == "weibull":
        mask_cens = Yi[:, 1] == 0
        return (
            pm.Weibull(
                f"{name}_nocen",
                alpha=shape,
                beta=pt.exp(pred[~mask_cens] / shape),
                observed=Yi[~mask_cens, 0],
            ),
            pm.Potential(
                f"{name}_censor",
                -(pt.pow(Yi[mask_cens, 0], shape) * pt.exp(-pred[mask_cens])),
            ),
        )
    elif kind == "piece-const":
        t, e = Yi[:, 0], Yi[:, 1].astype(int)
        interval_lens = interval_bounds[1:] - interval_bounds[:-1]
        n_intervals = len(interval_lens)
        mask = (t[:, None] > interval_bounds[:-1]) & (
            t[:, None] <= interval_bounds[1:]
        )
        y_ind = np.nonzero(mask)[1]
        # 需要构造一个n_samples x n_intervals的0-1矩阵,
        # 当sample i的event在第j个interval中出现时,该矩阵的ij元=1,其他为0.
        e_mat = np.zeros((len(t), n_intervals))
        e_mat[np.arange(len(t)), y_ind] = e
        # 需要构造一个n_samples x n_intervals的正实数矩阵, 当sample i的
        # event或censor发生在第j个interval中出现时, i1,...,i(j-1)的元素
        # 为对应的interval长度,ij的元素是t-第j个interval的begin, i(j+1),...是0
        t_mat = (t[:, None] >= interval_bounds[:-1]) * interval_lens
        t_mat[np.arange(len(t)), y_ind] = t - interval_bounds[y_ind]
        return pm.Poisson(
            name, pt.exp(pred)[:, None] * lambda0 * t_mat, observed=e_mat
        )


class BBP(BiomarkerPoolBase):

    def __init__(
        self,
        prior_betax_std: TYPE_STD = 10,
        prior_betaz_std: TYPE_STD = 10,
        prior_mu_x_std: TYPE_STD = 10.0,
        prior_std_x_scale: TYPE_STD = 1.0,
        prior_y_scale: TYPE_STD = 1.0,
        prior_alpha_scale: TYPE_STD = 1.0,
        prior_mu_a_std: TYPE_STD = 10.0,
        prior_std_a_scale: TYPE_STD = 1.0,
        prior_mu_b_std: TYPE_STD = 10.0,
        prior_std_b_scale: TYPE_STD = 1.0,
        prior_mu_beta0_std: TYPE_STD = 10.0,
        prior_std_beta0_scale: TYPE_STD = 1.0,
        solver: Literal["pymc", "blackjax"] = "pymc",
        nsample: int = 1000,
        ntunes: int = 1000,
        nchains: int = 4,
        pbar: bool = True,
        seed: int = 0,
        target_accept: Optional[float] = 0.99,
        type_outcome: Literal["binary", "continue", "survival"] = "binary",
        multiple_imputation: bool = False,
    ) -> None:
        """
        calibration_approximation=True means that P(W|X, Z) = P(W|X)
        """

        assert prior_betax_std == "inf" or (
            isinstance(prior_betax_std, float) and prior_betax_std > 0
        )
        assert prior_betaz_std == "inf" or (
            isinstance(prior_betaz_std, float) and prior_betaz_std > 0
        )
        assert solver in ["pymc", "vi", "blackjax"]
        assert type_outcome in ["binary", "continue", "survival"]

        self.prior_betax_std_ = prior_betax_std
        self.prior_betaz_std_ = prior_betaz_std
        self.prior_mu_x_std_ = prior_mu_x_std
        self.prior_std_x_scale_ = prior_std_x_scale
        self.prior_y_scale_ = prior_y_scale
        self.prior_alpha_scale_ = prior_alpha_scale
        self.prior_mu_a_std_ = prior_mu_a_std
        self.prior_std_a_scale_ = prior_std_a_scale
        self.prior_mu_b_std_ = prior_mu_b_std
        self.prior_std_b_scale_ = prior_std_b_scale
        self.prior_mu_beta0_std_ = prior_mu_beta0_std
        self.prior_std_beta0_scale_ = prior_std_beta0_scale

        self.solver_ = solver
        self.nsample_ = nsample
        self.ntunes_ = ntunes
        self.nchains_ = nchains
        self.pbar_ = pbar
        self.seed_ = seed
        self.target_accept_ = target_accept
        self.type_outcome_ = type_outcome
        self.multi_imp_ = multiple_imputation

    def fit(
        self,
        df: pd.DataFrame,
        X_col: str = "X",
        S_col: str = "S",
        W_col: str = "W",
        Y_col: Union[str, Tuple[str, str]] = "Y",
        Z_col: Optional[Union[str, Sequence[str]]] = None,
    ) -> BBPResults:

        if self.type_outcome_ == "survival":
            assert isinstance(Y_col, (list, tuple)) and len(Y_col) == 2
            # T, E

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
            # 1.1 == mu_x, sigma_x ==
            mu_x = set_real_value_rv("mu_x", self.prior_mu_x_std_)
            sigma_x = set_pos_value_rv("sigma_x", self.prior_std_x_scale_)
            # convert the mean-sigma format to alpha-beta format
            # beta = (
            #     self.mean_prior_sigma_x_
            #     + np.sqrt(
            #         self.mean_prior_sigma_x_ + self.sigma_prior_sigma_x_**2
            #     )
            # ) / (2 * self.sigma_prior_sigma_x_**2)
            # alpha = self.mean_prior_sigma_x_ * beta + 1
            # sigma_x = pm.Gamma("sigma_x", alpha=alpha, beta=beta)
            # 1.2 == betax ==
            betax = set_real_value_rv("betax", self.prior_betax_std_)
            # 1.3 == beta_z ==
            if Z is not None:
                beta_z = set_real_value_rv(
                    "betaz", self.prior_betaz_std_, dims="Z"
                )
            # 1.4 == scale of y (continue or survival) ==
            if self.type_outcome_ == "continue":
                sigma_y = set_pos_value_rv("sigma_y", self.prior_y_scale_)
            elif self.type_outcome_ == "survival":
                alpha_surv = set_pos_value_rv(
                    "alpha_surv", self.prior_alpha_scale_
                )
            # 1.5 == sigma_w_s ==
            sigma_ws_scale = pm.HalfCauchy("sigma_ws_scale", 1.0)
            sigma_ws_tilde = pm.Uniform("sigma_ws_tilde", 0, 1, dims="S")
            sigma_ws = pm.Deterministic(
                "sigma_ws",
                pt.tan(np.pi / 2 * sigma_ws_tilde) * sigma_ws_scale,
                dims="S",
            )
            # mu_sigma_w = pm.HalfFlat("mu_sigma_w")  # mu_sigma_w这里其实是mode
            # sigma_sigma_w = pm.HalfCauchy("sigma_sigma_w", 1.0)
            # if self.prior_sigma_ws_ == "gamma":
            #     sigma_ws = pm.Gamma(
            #         "sigma_ws", mu=mu_sigma_w, sigma=sigma_sigma_w, dims="S"
            #     )
            # elif self.prior_sigma_ws_ == "inv_gamma":
            #     sigma2_ws = pm.InverseGamma(
            #         "sigma2_ws", mu=mu_sigma_w, sigma=sigma_sigma_w, dims="S"
            #     )
            #     sigma_ws = pm.Deterministic(
            #         "sigma_ws", pm.math.sqrt(sigma2_ws), dims="S"
            #     )
            # 1.6 == a_s, b_s, beta0_s ==
            # reparameterization to reduce divergences
            mu_a = set_real_value_rv("mu_a", self.prior_mu_a_std_)
            std_a = set_pos_value_rv("std_a", self.prior_std_a_scale_)
            a_error = pm.Normal("a_error", 0, 1, dims="S")
            a_s = pm.Deterministic("a_s", std_a * a_error + mu_a)

            mu_b = set_real_value_rv("mu_b", self.prior_mu_b_std_)
            std_b = set_pos_value_rv("std_b", self.prior_std_b_scale_)
            b_error = pm.Normal("b_error", 0, 1, dims="S")
            b_s = pm.Deterministic("b_s", std_b * b_error + mu_b)

            mu_beta0 = set_real_value_rv("mu_beta0", self.prior_mu_beta0_std_)
            std_beta0 = set_pos_value_rv(
                "std_beta0", self.prior_std_beta0_scale_
            )
            beta0_error = pm.Normal("beta0_error", 0, 1, dims="S")
            beta0s = pm.Deterministic(
                "beta0_s", std_beta0 * beta0_error + mu_beta0
            )

            # ============= 2. set data generation model =============
            # for samples that can not see X
            # reparameterization to reduce divergences
            X_no_obs_error = pm.Normal("X_no_obs_error", 0, 1, size=n_xUnKnow)
            X_no_obs = pm.Deterministic(
                "X_no_obs", X_no_obs_error * sigma_x + mu_x
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
                if self.type_outcome_ == "binary":
                    pm.Bernoulli(
                        f"Y_{si}_no_obs_X",
                        logit_p=pred,
                        observed=Yi,
                    )
                elif self.type_outcome_ == "continue":
                    pm.Normal(
                        f"Y_{si}_no_obs_X",
                        pred,
                        sigma_y,
                        observed=Yi,
                    )
                elif self.type_outcome_ == "survival":
                    set_surv_likelihood(
                        f"Y_{si}_no_obs_X",
                        Yi=Yi,
                        pred=pred,
                        kind="weibull",
                        shape=alpha_surv,
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
                if self.type_outcome_ == "binary":
                    pm.Bernoulli(
                        f"Y_{si}_obs_X",
                        logit_p=pred,
                        observed=Yi,
                    )
                elif self.type_outcome_ == "continue":
                    pm.Normal(
                        f"Y_{si}_obs_X",
                        pred,
                        sigma_y,
                        observed=Yi,
                    )
                elif self.type_outcome_ == "survival":
                    set_surv_likelihood(
                        f"Y_{si}_obs_X",
                        Yi=Yi,
                        pred=pred,
                        kind="weibull",
                        shape=alpha_surv,
                    )

            # ============= 3. MCMC sampling =============
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

        if self.multi_imp_:
            X_on_obs_val = res.posterior.X_no_obs.values.reshape(-1, n_xUnKnow)
            all_beta = []
            for i in tqdm(
                range(self.nchains_ * self.nsample_),
                disable=not self.pbar_,
                desc="Multiple Imputation: ",
            ):
                X_imp = X.copy()
                X_imp[is_m] = X_on_obs_val[i, :]
                X_imp = X_imp[:, None]
                if Z is not None:
                    X_imp = np.concatenate([X_imp, Z], axis=1)
                if self.type_outcome_ == "continue":
                    estimator = LinearRegression()
                    estimator.fit(X_imp, Y)
                    beta = estimator.coef_
                elif self.type_outcome_ == "binary":
                    estimator = LogisticRegression()
                    estimator.fit(X_imp, Y)
                    beta = estimator.coef_
                elif self.type_outcome_ == "survival":
                    df_imp = pd.DataFrame(X_imp)
                    df_imp.columns = (
                        ["X"] if Z is None else ["X"] + list(Z_col)
                    )
                    df_imp["T"] = Y[:, 0]
                    df_imp["E"] = Y[:, 1].astype(int)
                    estimator = CoxPHFitter()
                    estimator.fit(df=df_imp, duration_col="T", event_col="E")
                    beta = estimator.hazards_

                all_beta.append(beta)
            all_beta = np.stack(all_beta, axis=0)
        else:
            all_beta = None

        return BBPResults(model, res, all_beta)
