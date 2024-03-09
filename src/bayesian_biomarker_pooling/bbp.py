import os
from typing import Literal, Optional, Tuple, Union, List

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


# 如果是2-tuple，则表示Normal($0, $1^2)
# 如果是3-tuple，则表示Normal(Normal($0, $1), HalfCauchy($2)^2)
# 如果是4-tuple，则表示Normal(Normal($0, $1), InverseGamma(alpha=$2, beta=$3)^2)
HYPER_PRIOR = (
    Tuple[float, float]
    | Tuple[float, float, float]
    | Tuple[float, float, float, float]
)
# 如果是float，则表示HalfCauchy($0)
# 如果是2-tuple，则表示InverseGamma(alpha=$0, beta=$1)
# 如果是4-tuple，则表示InverseGamma(alpha=Gamma($0, $1), beta=Gamma($2, $3))
SIGMA_PRIOR = (
    float
    | Tuple[float, float]
    | Tuple[float, float, float]
    | Tuple[float, float, float, float]
)


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


def set_hyper_prior(
    prior_params: HYPER_PRIOR, n_studies: int, name: str
) -> pm.Normal:
    assert isinstance(prior_params, tuple) and len(prior_params) in (2, 3, 4)
    for i in prior_params[1:]:
        assert i > 0.0

    if len(prior_params) == 2:
        return pm.Normal(
            name, mu=prior_params[0], sigma=prior_params[1], size=n_studies
        )

    hyper_mu = pm.Normal(
        f"mu_{name}", mu=prior_params[0], sigma=prior_params[1]
    )
    if len(prior_params) == 3:
        hyper_sigma = pm.HalfCauchy(f"sigma_{name}", prior_params[2])
        return pm.Normal(name, mu=hyper_mu, sigma=hyper_sigma, size=n_studies)

    hyper_sigma = pm.InverseGamma(
        f"sigma_{name}", alpha=prior_params[2], beta=prior_params[3]
    )
    return pm.Normal(name, mu=hyper_mu, sigma=hyper_sigma, size=n_studies)


def set_sigma_prior(
    prior_params: SIGMA_PRIOR, n_studies: int, name: str
) -> pm.InverseGamma:
    assert isinstance(prior_params, float) or (
        isinstance(prior_params, tuple) and len(prior_params) in (1, 2, 4)
    )
    if isinstance(prior_params, tuple):
        for i in prior_params:
            assert i > 0.0

    if isinstance(prior_params, float):
        return pm.HalfCauchy(name, prior_params, size=n_studies)
    if len(prior_params) == 2:
        return pm.InverseGamma(
            name, alpha=prior_params[0], beta=prior_params[1], size=n_studies
        )
    if len(prior_params) == 4:
        hyper_alpha = pm.Gamma(
            f"alpha_{name}", alpha=prior_params[0], beta=prior_params[1]
        )
        hyper_beta = pm.Gamma(
            f"beta_{name}", alpha=prior_params[2], beta=prior_params[3]
        )
        return pm.InverseGamma(
            name, alpha=hyper_alpha, beta=hyper_beta, size=n_studies
        )


def create_model(
    n_studies: int,
    n_xKnow: int,
    n_xUnKnow: int,
    X_Know: np.ndarray,
    XWY_xKnow: List[Tuple[np.ndarray, np.ndarray, np.ndarray] | None],
    WY_xUnKnow: List[Tuple[np.ndarray, np.ndarray, np.ndarray] | None],
    prior_betax: Tuple[float, float] = (0.0, 1.0),
    prior_x: HYPER_PRIOR | None = None,
    prior_sigma: SIGMA_PRIOR = 1.0,
    prior_a: HYPER_PRIOR = (0.0, 1.0, 1.0),
    prior_b: HYPER_PRIOR = (0.0, 1.0, 1.0),
    prior_beta0: HYPER_PRIOR = (0.0, 1.0, 1.0),
):

    # TODO: assert prior_x

    with pm.Model() as model:

        betax = pm.Normal("betax", mu=prior_betax[0], sigma=prior_betax[1])
        sigma_ws = set_sigma_prior(prior_sigma, n_studies, "sigma_ws")
        a_s = set_hyper_prior(prior_a, n_studies, "a")
        b_s = set_hyper_prior(prior_b, n_studies, "b")
        beta0s = set_hyper_prior(prior_beta0, n_studies, "beta0")

        if prior_x is None:
            mu_x, sigma_x = X_Know.mean(), X_Know.std()
        elif len(prior_x) == 2:
            mu_x, sigma_x = prior_x
        elif len(prior_x) == 3:
            mu_x = pm.Normal("mu_x", prior_x[0], prior_x[1])
            sigma_x = pm.HalfCauchy("sigma_x", prior_x[2])
            pm.Normal("X_obs", mu_x, sigma_x, observed=X_Know)
        elif len(prior_x) == 4:
            mu_x = pm.Normal("mu_x", prior_x[0], prior_x[1])
            sigma_x = pm.InverseGamma(
                "sigma_x", alpha=prior_x[2], beta=prior_x[3]
            )
        X_no_obs = pm.Normal(
            "X_no_obs",
            mu_x,
            sigma_x,
            size=n_xUnKnow,
        )

        # for samples that can not see X
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
        prior_betax: Tuple[float, float] = (0.0, 10.0),
        prior_x: HYPER_PRIOR | None = (0, 10.0, 1.0),
        prior_sigma: float = 1.0,
        prior_a: HYPER_PRIOR = (0.0, 10.0, 1.0),
        prior_b: HYPER_PRIOR = (0.0, 10.0, 1.0),
        prior_beta0: HYPER_PRIOR = (0.0, 10.0, 1.0),
        solver: Literal["pymc", "vi"] = "pymc",
        nsample: int = 1000,
        ntunes: int = 1000,
        nchains: int = 4,
        pbar: bool = True,
        seed: int = 0,
    ) -> None:

        assert (
            isinstance(prior_betax, tuple)
            and len(prior_betax) == 2
            and prior_betax[1] > 0.0
        )

        self.prior_betax_ = prior_betax
        self.prior_x_ = prior_x
        self.prior_sigma_log_ = prior_sigma
        self.prior_a_ = prior_a
        self.prior_b_ = prior_b
        self.prior_beta0_ = prior_beta0
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
            n_studies=n_studies,
            n_xKnow=n_xKnow,
            n_xUnKnow=n_xUnKnow,
            X_Know=X_Know,
            XWY_xKnow=XWY_xKnow,
            WY_xUnKnow=WY_xUnKnow,
            prior_betax=self.prior_betax_,
            prior_x=self.prior_x_,
            prior_sigma=self.prior_sigma_log_,
            prior_a=self.prior_a_,
            prior_b=self.prior_b_,
            prior_beta0=self.prior_beta0_,
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

    def plot(
        self, save_root: str, var_names: Optional[Tuple[str]] = ("betax",)
    ):
        os.makedirs(save_root, exist_ok=True)

        axes = az.plot_trace(self.res_, var_names=var_names, show=False)
        if isinstance(axes, np.ndarray):
            fig = axes.flatten()[0].figure
        else:
            fig = axes.figure
        fig.savefig(os.path.join(save_root, "trace.png"))

        axes = az.plot_density(self.res_, var_names=var_names, show=False)
        if isinstance(axes, np.ndarray):
            fig = axes.flatten()[0].figure
        else:
            fig = axes.figure
        fig.savefig(os.path.join(save_root, "density.png"))

        axes = az.plot_posterior(self.res_, var_names=var_names, show=False)
        if isinstance(axes, np.ndarray):
            fig = axes.flatten()[0].figure
        else:
            fig = axes.figure
        fig.savefig(os.path.join(save_root, "posterior.png"))
