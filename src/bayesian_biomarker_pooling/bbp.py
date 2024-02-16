from typing import Literal, Optional, Tuple, Union, List

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


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
]


def set_prior(
    n_studies: int,
    prior_betax: Literal["flat", "standard_normal"],
    prior_sigma_ws: Literal["gamma", "inv_gamma"],
    prior_sigma_ab0: Literal["half_cauchy", "half_flat"] = "half_cauchy",
    std_prior_a: float = 1.0,
    std_prior_b: float = 1.0,
    std_prior_beta0: float = 1.0,
) -> Tuple[pm.Distribution]:

    assert prior_betax in ["flat", "standard_normal"]
    assert prior_sigma_ws in ["gamma", "inv_gamma"]
    assert prior_sigma_ab0 in ["half_cauchy", "half_flat"]

    betax = (
        pm.Flat("betax")
        if prior_betax == "flat"
        else pm.Normal("betax", mu=0, sigma=1)
    )

    mu_sigma_w = pm.HalfFlat("mu_sigma_w")  # mu_sigma_w这里其实是mode
    sigma_sigma_w = pm.HalfCauchy("sigma_sigma_w", 1.0)
    if prior_sigma_ws == "gamma":
        sigma_ws = pm.Gamma(
            "sigma_ws", mu=mu_sigma_w, sigma=sigma_sigma_w, size=n_studies
        )
    elif prior_sigma_ws == "inv_gamma":
        sigma2_ws = pm.InverseGamma(
            "sigma2_ws",
            mu=mu_sigma_w,
            sigma=sigma_sigma_w,
            size=n_studies,
        )
        sigma_ws = pm.Deterministic("sigma_ws", pm.math.sqrt(sigma2_ws))

    a = pm.Normal("a", 0, std_prior_a)
    b = pm.Normal("b", 0, std_prior_b)
    beta0 = pm.Normal("beta0", 0, std_prior_beta0)
    if prior_sigma_ab0 == "half_cauchy":
        sigma_a = pm.HalfCauchy("sigma_a", 1.0)
        sigma_b = pm.HalfCauchy("sigma_b", 1.0)
        sigma_0 = pm.HalfCauchy("sigma_0", 1.0)
    elif prior_sigma_ab0 == "half_flat":
        sigma_a = pm.HalfFlat("sigma_a")
        sigma_b = pm.HalfFlat("sigma_b")
        sigma_0 = pm.HalfFlat("sigma_0")
    a_s = pm.Normal("a_s", a, sigma_a, size=n_studies)
    b_s = pm.Normal("b_s", b, sigma_b, size=n_studies)
    beta0s = pm.Normal("beta0s", beta0, sigma_0, size=n_studies)

    return a_s, b_s, sigma_ws, beta0s, betax


def create_model(
    n_studies: int,
    n_xKnow: int,
    n_xUnKnow: int,
    X_Know: np.ndarray,
    XWY_xKnow: List[Tuple[np.ndarray, np.ndarray, np.ndarray] | None],
    WY_xUnKnow: List[Tuple[np.ndarray, np.ndarray, np.ndarray] | None],
    prior_betax: Literal["flat", "standard_normal"] = "standard_normal",
    prior_sigma_ws: Literal["gamma", "inv_gamma"] = "gamma",
    prior_sigma_ab0: Literal["half_cauchy", "half_flat"] = "half_cauchy",
    std_prior_a: float = 1.0,
    std_prior_b: float = 1.0,
    std_prior_beta0: float = 1.0,
    std_prior_mu_x: float = 10.0,
    mean_prior_sigma_x: float = 1.0,
    sigma_prior_sigma_x: float = 1.0,
):

    with pm.Model() as model:
        (
            a_s,
            b_s,
            sigma_ws,
            beta0s,
            betax,
        ) = set_prior(
            n_studies,
            prior_betax,
            prior_sigma_ws,
            prior_sigma_ab0,
            std_prior_a,
            std_prior_b,
            std_prior_beta0,
        )

        mu_x = pm.Normal("mu_x", 0, std_prior_mu_x)
        beta = (
            mean_prior_sigma_x
            + np.sqrt(mean_prior_sigma_x + sigma_prior_sigma_x**2)
        ) / (2 * sigma_prior_sigma_x**2)
        alpha = mean_prior_sigma_x * beta + 1
        sigma_x = pm.Gamma("sigma_x", alpha=alpha, beta=beta)

        if beta0s.ndim == 0:
            beta0s = [beta0s] * n_studies
        if sigma_ws.ndim == 0:
            sigma_ws = [sigma_ws] * n_studies

            # for samples that can not see X
            X_no_obs = pm.Normal(
                "X_no_obs",
                mu_x,
                sigma_x,
                size=n_xUnKnow,
            )
            start = 0
            for i, WY in enumerate(WY_xUnKnow):
                if WY is None:
                    continue
                W, Y = WY
                end = start + W.shape[0]
                X_no_obs_i = X_no_obs[start:end]
                pm.Normal(
                    "W_%d_no_obs_X" % i,
                    a_s[i] + b_s[i] * X_no_obs_i,
                    sigma_ws[i],
                    observed=W,
                )
                pm.Bernoulli(
                    "Y_%d_no_obs_X" % i,
                    logit_p=beta0s[i] + betax * X_no_obs_i,
                    observed=Y,
                )
                start = end

            # for samples that can see X
            pm.Normal("X_obs", mu_x, sigma_x, observed=X_Know)
            for i, XWY in enumerate(XWY_xKnow):
                if XWY is None:
                    continue
                X, W, Y = XWY
                pm.Normal(
                    "W_%d_obs_X" % i,
                    a_s[i] + b_s[i] * X,
                    sigma_ws[i],
                    observed=W,
                )
                pm.Bernoulli(
                    "Y_%d_obs_X" % i,
                    logit_p=beta0s[i] + betax * X,
                    observed=Y,
                )

    return model


def train_model(
    model: pm.Model,
    solver: Literal["pymc", "vi"] = "pymc",
    nsample: int = 1000,
    ntunes: int = 1000,
    nchains: int = 4,
    pbar: bool = True,
    seed: Optional[int] = None,
) -> Union[pm.Approximation, az.InferenceData]:
    with model:
        if solver == "vi":
            res = pm.fit(progressbar=pbar, random_seed=seed)
        else:
            res = pm.sample(
                nsample,
                tune=ntunes,
                chains=nchains,
                cores=nchains,
                progressbar=pbar,
                random_seed=list(range(seed, seed + nchains)),
                nuts_sampler=solver,
            )
    return res


def summary_res(
    res: Union[pm.Approximation, az.InferenceData],
    solver: Literal["pymc", "vi"] = "pymc",
    var_names: Optional[Tuple[str]] = ("betax",),
    return_obj: Literal["raw", "point_interval"] = "point_interval",
) -> pd.DataFrame:
    str_all_hidden_variables = ",".join(all_hidden_variables)
    assert (
        len(set(var_names).difference(all_hidden_variables)) == 0
    ), f"The element of var_names must be one of [{str_all_hidden_variables}]"
    assert return_obj in ["raw", "point_interval"]

    if return_obj == "raw":
        return res

    if solver != "vi":
        res_df = az.summary(
            res,
            hdi_prob=0.95,
            kind="stats",
            var_names=list(var_names),
        )
    else:
        # TODO: 有一些param是log__之后的
        nparam = res.ndim
        param_names = [None] * nparam
        # log_slices = []
        if var_names is not None:
            post_mu = res.mean.eval()
            post_sd = res.std.eval()
            res_df = []
            for vari in var_names:
                if vari in res.ordering:
                    pass
                elif (vari + "_log__") in res.ordering:
                    vari = vari + "_log__"
                else:
                    raise KeyError(
                        "%s or %s_log__ not in approximation.ordering."
                        % (vari, vari)
                    )
                slice_ = res.ordering[vari][1]
                post_mu_i = post_mu[slice_]
                post_sd_i = post_sd[slice_]
                n = len(post_mu_i)
                res_df.append(
                    pd.DataFrame(
                        {"mean": post_mu_i, "sd": post_sd_i},
                        index=(
                            [vari]
                            if n == 1
                            else ["%s[%d]" % (vari, i) for i in range(n)]
                        ),
                    )
                )
            res_df = pd.concat(res_df)
        else:
            for param, (_, slice_, _, _) in res.ordering.items():
                if slice_.step is not None:
                    raise NotImplementedError
                if slice_.start >= nparam:
                    continue
                elif slice_.stop > nparam:
                    slice_ = slice(slice_.start, nparam)
                n = slice_.stop - slice_.start
                # if param.endswith("_log__"):
                #     param = param[:-6]
                #     log_slices.append(slice_)
                if n > 1:
                    param_names[slice_] = [
                        "%s[%d]" % (param, i) for i in range(n)
                    ]
                else:
                    param_names[slice_] = [param] * n
            res_df = pd.DataFrame(
                {
                    "mean": res.mean.eval(),
                    "sd": res.std.eval(),
                },
                index=param_names,
            )
        res_df["hdi_2.5%"] = res_df["mean"] - 1.96 * res_df["sd"]
        res_df["hdi_97.5%"] = res_df["mean"] + 1.96 * res_df["sd"]

    return res_df


class BBP:

    def __init__(
        self,
        prior_betax: Literal["flat", "standard_normal"] = "standard_normal",
        prior_sigma_ws: Literal["gamma", "inv_gamma"] = "gamma",
        prior_sigma_ab0: Literal["half_cauchy", "half_flat"] = "half_cauchy",
        std_prior_a: float = 1.0,
        std_prior_b: float = 1.0,
        std_prior_beta0: float = 1.0,
        std_prior_mu_x: float = 10.0,
        mean_prior_sigma_x: float = 1.0,
        sigma_prior_sigma_x: float = 1.0,
        solver: Literal["pymc", "vi"] = "pymc",
        nsample: int = 1000,
        ntunes: int = 1000,
        nchains: int = 4,
        pbar: bool = True,
        seed: int = 0,
    ) -> None:

        assert prior_betax in ["flat", "standard_normal"]
        assert prior_sigma_ws in ["gamma", "inv_gamma"]
        assert prior_sigma_ab0 in ["half_cauchy", "half_flat"]
        assert solver in ["pymc", "vi"]

        self.prior_betax_ = prior_betax
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

    def fit(
        self,
        df: pd.DataFrame,
        X_col: str = "X",
        S_col: str = "S",
        W_col: str = "W",
        Y_col: str = "Y",
    ) -> None:

        # 把dataframe都处理好后送入create_model方法来创建pm.Model
        ind_studies = df[S_col].unique()
        ind_xKnow = df[X_col].notna()
        ind_xUnKnow = df[X_col].isna()
        n_studies = int(ind_studies.shape[0])
        n_xKnow = int(ind_xKnow.sum())
        n_xUnKnow = int(ind_xUnKnow.sum())

        X_Know = df.loc[ind_xKnow, X_col].values

        XWY_xKnow, WY_xUnKnow = [], []
        for si in ind_studies:
            dfi = df.loc[(df[S_col] == si) & (ind_xKnow), :]
            if dfi.shape[0] == 0:
                XWY_xKnow.append(None)
            else:
                XWY_xKnow.append(
                    (dfi[X_col].values, dfi[W_col].values, dfi[Y_col].values)
                )
            dfi = df.loc[(df[S_col] == si) & (ind_xUnKnow), :]
            if dfi.shape[0] == 0:
                WY_xUnKnow.append(None)
            else:
                WY_xUnKnow.append((dfi[W_col].values, dfi[Y_col].values))

        self.model_ = create_model(
            n_studies,
            n_xKnow,
            n_xUnKnow,
            X_Know,
            XWY_xKnow,
            WY_xUnKnow,
            self.prior_betax_,
            self.prior_sigma_ws_,
            self.prior_sigma_ab0_,
            self.std_prior_a_,
            self.std_prior_b_,
            self.std_prior_beta0_,
            self.std_prior_mu_x_,
            self.mean_prior_sigma_x_,
            self.sigma_prior_sigma_x_,
        )

        self.res_ = train_model(
            self.model_,
            self.solver_,
            self.nsample_,
            self.ntunes_,
            self.nchains_,
            self.pbar_,
            self.seed_,
        )

    def summary(
        self,
        var_names: Optional[Tuple[str]] = ("betax",),
        return_obj: Literal["raw", "point_interval"] = "point_interval",
    ) -> Union[pm.Approximation, az.InferenceData, pd.DataFrame]:

        return summary_res(self.res_, self.solver_, var_names, return_obj)
