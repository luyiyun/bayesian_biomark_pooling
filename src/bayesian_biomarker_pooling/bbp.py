import os
from typing import Literal, Optional, Tuple, Union, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


# 如果是2-tuple，则表示Normal($0, $1^2)
# 如果是3-tuple，则表示Normal(Normal($0, $1), HalfCauchy($2)^2)
# 如果是4-tuple，则表示Normal(Normal($0, $1), InverseGamma(alpha=$2, beta=$3)^2)
HYPER_PRIOR_DIST = Literal[
    "normal", "normal-normal-halfcauchy", "normal-normal-invgamma"
]
HYPER_PRIOR_ARGS = (
    Tuple[float, float] | Tuple[float, float, float] | Tuple[float, float, float, float]
)
# 如果是float，则表示HalfCauchy($0)
# 如果是2-tuple，则表示InverseGamma(alpha=$0, beta=$1)
# 如果是4-tuple，则表示InverseGamma(alpha=Gamma($0, $1), beta=Gamma($2, $3))
SIGMA_PRIOR_DIST = Literal["halfcauchy", "invgamma", "invgamma-gamma-gamma"]
SIGMA_PRIOR_ARGS = (
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
    "betaz",
]


def set_hyper_prior(
    prior_dist: HYPER_PRIOR_DIST,
    prior_args: HYPER_PRIOR_ARGS,
    n_studies: int | None,
    name: str,
    obs: np.ndarray | None = None,
) -> Tuple[
    pm.Distribution, Union[pm.Distribution, float], Union[pm.Distribution, float]
]:
    if prior_dist == "normal":
        dist = pm.Normal(
            name, mu=prior_args[0], sigma=prior_args[1], size=n_studies, observed=obs
        )
        return dist, *prior_args

    hyper_mu = pm.Normal(
        f"mu_{name}",
        mu=prior_args[0],
        sigma=prior_args[1],
    )
    if prior_dist == "normal-normal-halfcauchy":
        hyper_sigma = pm.HalfCauchy(f"sigma_{name}", prior_args[2])
        dist = pm.Normal(
            name, mu=hyper_mu, sigma=hyper_sigma, size=n_studies, observed=obs
        )
    elif prior_dist == "normal-normal-invgamma":
        hyper_sigma = pm.InverseGamma(
            f"sigma_{name}", alpha=prior_args[2], beta=prior_args[3]
        )
        dist = pm.Normal(
            name, mu=hyper_mu, sigma=hyper_sigma, size=n_studies, observed=obs
        )
    return dist, hyper_mu, hyper_sigma


def set_sigma_prior(
    prior_dist: SIGMA_PRIOR_DIST,
    prior_args: SIGMA_PRIOR_ARGS,
    n_studies: int,
    name: str,
    obs: np.ndarray | None = None,
) -> pm.InverseGamma:
    if prior_dist == "halfcauchy":
        return pm.HalfCauchy(name, prior_args, size=n_studies, observed=obs)
    elif prior_dist == "invgamma":
        return pm.InverseGamma(
            name, alpha=prior_args[0], beta=prior_args[1], size=n_studies, observed=obs
        )
    elif prior_dist == "invgamma-gamma-gamma":
        hyper_alpha = pm.Gamma(f"alpha_{name}", alpha=prior_args[0], beta=prior_args[1])
        hyper_beta = pm.Gamma(f"beta_{name}", alpha=prior_args[2], beta=prior_args[3])
        return pm.InverseGamma(
            name, alpha=hyper_alpha, beta=hyper_beta, size=n_studies, observed=obs
        )


@dataclass
class BBP:
    prior_betax_args: Tuple[float, float] = (0.0, 10.0)
    prior_x_dist: HYPER_PRIOR_DIST = "normal-normal-halfcauchy"
    prior_x_args: HYPER_PRIOR_ARGS | None = (0, 10.0, 1.0)
    prior_sigma_dist: SIGMA_PRIOR_DIST = "halfcauchy"
    prior_sigma_args: SIGMA_PRIOR_ARGS = 1.0
    prior_a_dist: HYPER_PRIOR_DIST = "normal-normal-halfcauchy"
    prior_a_args: HYPER_PRIOR_ARGS = (0.0, 10.0, 1.0)
    prior_b_dist: HYPER_PRIOR_DIST = "normal-normal-halfcauchy"
    prior_b_args: HYPER_PRIOR_ARGS = (0.0, 10.0, 1.0)
    prior_beta0_dist: HYPER_PRIOR_DIST = "normal-normal-halfcauchy"
    prior_beta0_args: HYPER_PRIOR_ARGS = (0.0, 10.0, 1.0)
    prior_betaz_args: Tuple[float, float] = (0.0, 10.0)
    solver: Literal["pymc", "vi", "blackjax"] = "pymc"
    nsample: int = 1000
    ntunes: int = 1000
    nchains: int = 4
    pbar: bool = True
    seed: int = 0

    def __post_init__(self):
        if self.solver == "blackjax":
            try:
                import blackjax
            except ImportError:
                raise ImportError("Please install blackjax to use blackjax as solver")
        for prior_name in ["x", "a", "b", "beta0"]:
            dist = getattr(self, f"prior_{prior_name}_dist")
            assert dist in [
                "normal",
                "normal-normal-halfcauchy",
                "normal-normal-invgamma",
            ], (
                f"prior_{prior_name}_dist must be one of 'normal', 'normal-normal-halfcauchy', 'normal-normal-invgamma'"
                f", but got {dist}"
            )
            args = getattr(self, f"prior_{prior_name}_args")
            if dist == "normal":
                if prior_name == "x":
                    assert args is None or len(args) == 2, (
                        f"prior_{prior_name}_args must be None or 2-tuple for normal distribution"
                    )
                else:
                    assert len(args) == 2, (
                        f"prior_{prior_name}_args must be 2-tuple for normal distribution"
                    )
            elif dist == "normal-normal-halfcauchy":
                assert len(args) == 3, (
                    f"prior_{prior_name}_args must be 3-tuple for normal distribution with normal-halfcauchy prior"
                )
            elif dist == "normal-normal-invgamma":
                assert len(args) == 4, (
                    f"prior_{prior_name}_args must be 4-tuple for normal distribution with normal-invgamma prior"
                )

        if self.prior_sigma_dist == "halfcauchy":
            assert isinstance(self.prior_sigma_args, float), (
                "prior_sigma_args must be float for halfcauchy distribution"
            )
        elif self.prior_sigma_dist == "invgamma":
            assert len(self.prior_sigma_args) == 2, (
                "prior_sigma_args must be 2-tuple for invgamma distribution"
            )
        elif self.prior_sigma_dist == "invgamma-gamma-gamma":
            assert len(self.prior_sigma_args) == 4, (
                "prior_sigma_args must be 4-tuple for invgamma distribution with gamma-gamma prior"
            )

    def _create_model(
        self,
        n_studies: int,
        n_xKnow: int,
        n_xUnKnow: int,
        X_Know: np.ndarray,
        XWY_xKnow: List[Tuple[np.ndarray, np.ndarray, np.ndarray] | None],
        WY_xUnKnow: List[Tuple[np.ndarray, np.ndarray, np.ndarray] | None],
    ):
        # TODO: assert prior_x
        with pm.Model() as model:
            betax = pm.Normal(
                "betax", mu=self.prior_betax_args[0], sigma=self.prior_betax_args[1]
            )
            sigma_ws = set_sigma_prior(
                self.prior_sigma_dist,
                self.prior_sigma_args,
                n_studies,
                "sigma_ws",
            )
            a_s, _, _ = set_hyper_prior(
                self.prior_a_dist, self.prior_a_args, n_studies, "a"
            )
            b_s, _, _ = set_hyper_prior(
                self.prior_b_dist, self.prior_b_args, n_studies, "b"
            )
            beta0s, _, _ = set_hyper_prior(
                self.prior_beta0_dist, self.prior_beta0_args, n_studies, "beta0"
            )
            if self.prior_x_dist == "normal" and self.prior_x_args is None:
                mu_x, sigma_x = X_Know.mean(), X_Know.std()
                pm.Normal("X_obs", mu_x, sigma_x, observed=X_Know)
            else:
                _, mu_x, sigma_x = set_hyper_prior(
                    self.prior_x_dist, self.prior_x_args, None, "X_obs", obs=X_Know
                )

            X_no_obs = pm.Normal(
                "X_no_obs",
                mu_x,
                sigma_x,
                size=n_xUnKnow,
            )

            if self.n_Z > 0:
                betaz = pm.Normal(
                    "betaz",
                    mu=self.prior_betaz_args[0],
                    sigma=self.prior_betaz_args[1],
                    size=self.n_Z,
                )

            # for samples that can not see X
            start = 0
            for i, WY in enumerate(WY_xUnKnow):
                if WY is None:
                    continue
                W, Y, Z = WY
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
                    logit_p=(beta0s[i] + betax * X_no_obs_i)
                    if self.n_Z == 0
                    else (beta0s[i] + betax * X_no_obs_i + pm.math.dot(Z, betaz)),
                    observed=Y,
                )
                start = end

            # for samples that can see X
            for i, XWY in enumerate(XWY_xKnow):
                if XWY is None:
                    continue
                X, W, Y, Z = XWY
                pm.Normal(
                    "W_%d_obs_X" % i,
                    a_s[i] + b_s[i] * X,
                    sigma_ws[i],
                    observed=W,
                )
                pm.Bernoulli(
                    "Y_%d_obs_X" % i,
                    logit_p=(beta0s[i] + betax * X)
                    if self.n_Z == 0
                    else (beta0s[i] + betax * X + pm.math.dot(Z, betaz)),
                    observed=Y,
                )

        return model

    def _train_model(self, **kwargs: dict) -> Union[pm.Approximation, az.InferenceData]:
        with self.model_:
            if self.solver == "vi":
                res = pm.fit(progressbar=self.pbar, random_seed=self.seed)
            else:
                res = pm.sample(
                    self.nsample,
                    tune=self.ntunes,
                    chains=self.nchains,
                    cores=self.nchains,
                    progressbar=self.pbar,
                    random_seed=self.seed,
                    nuts_sampler=self.solver,
                    **kwargs,
                )
        return res

    def fit(
        self,
        df: pd.DataFrame,
        X_col: str = "X",
        S_col: str = "S",
        W_col: str = "W",
        Y_col: str = "Y",
        Z_col: List[str] | None = None,
        **kwargs: dict,
    ) -> None:
        self.n_Z = len(Z_col) if Z_col is not None else 0

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
                    (
                        dfi[X_col].values,
                        dfi[W_col].values,
                        dfi[Y_col].values,
                        None if self.n_Z == 0 else dfi[Z_col].values,
                    )
                )
            dfi = df.loc[(df[S_col] == si) & (ind_xUnKnow), :]
            if dfi.shape[0] == 0:
                WY_xUnKnow.append(None)
            else:
                WY_xUnKnow.append(
                    (
                        dfi[W_col].values,
                        dfi[Y_col].values,
                        None if self.n_Z == 0 else dfi[Z_col].values,
                    )
                )

        self.model_ = self._create_model(
            n_studies=n_studies,
            n_xKnow=n_xKnow,
            n_xUnKnow=n_xUnKnow,
            X_Know=X_Know,
            XWY_xKnow=XWY_xKnow,
            WY_xUnKnow=WY_xUnKnow,
        )

        self.res_ = self._train_model(**kwargs)

    def summary(
        self,
        var_names: Optional[Tuple[str]] = ("betax",),
        return_obj: Literal["raw", "point_interval"] = "point_interval",
    ) -> Union[pm.Approximation, az.InferenceData, pd.DataFrame]:
        str_all_hidden_variables = ",".join(all_hidden_variables)
        assert len(set(var_names).difference(all_hidden_variables)) == 0, (
            f"The element of var_names must be one of [{str_all_hidden_variables}]"
        )
        assert return_obj in ["raw", "point_interval"]

        if return_obj == "raw":
            return self.res_

        if self.solver != "vi":
            res_df = az.summary(
                self.res_,
                hdi_prob=0.95,
                kind="stats",
                var_names=list(var_names),
            )
        else:
            # TODO: 有一些param是log__之后的
            nparam = self.res_.ndim
            param_names = [None] * nparam
            # log_slices = []
            if var_names is not None:
                post_mu = self.res_.mean.eval()
                post_sd = self.res_.std.eval()
                res_df = []
                for vari in var_names:
                    if vari in self.res_.ordering:
                        pass
                    elif (vari + "_log__") in self.res_.ordering:
                        vari = vari + "_log__"
                    else:
                        raise KeyError(
                            "%s or %s_log__ not in approximation.ordering."
                            % (vari, vari)
                        )
                    slice_ = self.res_.ordering[vari][1]
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
                for param, (_, slice_, _, _) in self.res_.ordering.items():
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
                        param_names[slice_] = ["%s[%d]" % (param, i) for i in range(n)]
                    else:
                        param_names[slice_] = [param] * n
                res_df = pd.DataFrame(
                    {
                        "mean": self.res_.mean.eval(),
                        "sd": self.res_.std.eval(),
                    },
                    index=param_names,
                )
            res_df["hdi_2.5%"] = res_df["mean"] - 1.96 * res_df["sd"]
            res_df["hdi_97.5%"] = res_df["mean"] + 1.96 * res_df["sd"]

        return res_df

    def plot(self, save_root: str, var_names: Optional[Tuple[str]] = ("betax",)):
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
