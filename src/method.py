from typing import Literal, Optional, Tuple, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


class Model:
    def __init__(
        self,
        nsample: int = 1000,
        ntunes: int = 1000,
        nchains: int = 1,
        pbar: bool = False,
        solver: Literal[
            "pymc", "blackjax", "numpyro", "nutpie", "vi"
        ] = "pymc",
        seed: Optional[int] = None,
        prior_sigma_ws: Literal["gamma", "inv_gamma"] = "gamma",
        prior_sigma_ab0: Literal["half_cauchy", "half_flat"] = "half_cauchy",
        hier_prior_on_x: bool = True
    ) -> None:
        assert solver in ["pymc", "blackjax", "numpyro", "nutpie", "vi"]
        assert prior_sigma_ws in ["gamma", "inv_gamma"]
        assert prior_sigma_ab0 in ["half_cauchy", "half_flat"]

        self._nsample = nsample
        self._ntunes = ntunes
        self._nchains = nchains
        self._pbar = pbar
        self._solver = solver
        self._seed = seed
        self._prior_sigma_ws = prior_sigma_ws
        self._prior_sigma_ab0 = prior_sigma_ab0
        self._hier_prior_on_x = hier_prior_on_x

    def fit(
        self,
        df: pd.DataFrame,
        X_col: str = "X",
        S_col: str = "S",
        W_col: str = "W",
        Y_col: str = "Y",
    ) -> None:
        if not self._hier_prior_on_x:
            mu_x, sigma_x = np.mean(df[X_col]), np.std(df[X_col])
        all_s = df[S_col].unique()
        ns = all_s.shape[0]
        ind_xKnow = df[X_col].notna()
        ind_xUnKnow = df[X_col].isna()

        with pm.Model() as self._model:
            # mu_sigma_w = pm.HalfFlat("mu_sigma_w")
            mu_sigma_w = pm.HalfCauchy("mu_sigma_w", 1.0)
            sigma_sigma_w = pm.HalfCauchy("sigma_sigma_w", 1.0)
            if self._prior_sigma_ws == "gamma":
                sigma_ws = pm.Gamma(
                    "sigma_ws",
                    mu=mu_sigma_w,
                    sigma=sigma_sigma_w,
                    size=ns,
                )
            elif self._prior_sigma_ws == "inv_gamma":
                sigma2_ws = pm.InverseGamma(
                    "sigma2_ws",
                    mu=mu_sigma_w,
                    sigma=sigma_sigma_w,
                    size=ns,
                )
                sigma_ws = pm.Deterministic(
                    "sigma_ws", pm.math.sqrt(sigma2_ws)
                )

            if self._prior_sigma_ab0 == "half_cauchy":
                sigma_a = pm.HalfCauchy("sigma_a", 1.0)
                sigma_b = pm.HalfCauchy("sigma_b", 1.0)
                sigma_0 = pm.HalfCauchy("sigma_0", 1.0)
            elif self._prior_sigma_ab0 == "half_flat":
                sigma_a = pm.HalfFlat("sigma_a")
                sigma_b = pm.HalfFlat("sigma_b")
                sigma_0 = pm.HalfFlat("sigma_0")
            if self._hier_prior_on_x:
                mu_x = pm.Normal("mu_x", 0, 10)
                sigma_x = pm.HalfCauchy("sigma_x", 1.0)

            a = pm.Normal("a", 0, 10)
            b = pm.Normal("b", 0, 10)
            a_s = pm.Normal("a_s", a, sigma_a, size=ns)
            b_s = pm.Normal("b_s", b, sigma_b, size=ns)

            beta0 = pm.Flat("beta0")
            beta0s = pm.Normal("beta0s", beta0, sigma_0, size=ns)
            betax = pm.Flat("betax")

            # for samples that can not see X
            X_no_obs = pm.Normal(
                "X_no_obs",
                mu_x,
                sigma_x,
                size=(ind_xUnKnow.sum(),),
            )
            start = 0
            for i, si in enumerate(all_s):
                dfi = df.loc[(df[S_col] == si) & (~ind_xUnKnow), :]
                if dfi.shape[0] == 0:
                    continue
                end = start + dfi.shape[0]
                X_no_obs_i = X_no_obs[start:end]
                pm.Normal(
                    "W_%s_no_obs_X" % str(si),
                    a_s[i] + b_s[i] * X_no_obs_i,
                    sigma_ws[i],
                    observed=dfi[W_col].values,
                )
                pm.Bernoulli(
                    "Y_%s_no_obs_X" % str(si),
                    logit_p=beta0s[i] + betax * X_no_obs_i,
                    observed=dfi[Y_col].values,
                )
                start = end

            # for samples that can see X
            if self._hier_prior_on_x:
                pm.Normal(
                    "X_obs",
                    mu_x,
                    sigma_x,
                    observed=df.loc[ind_xKnow, X_col].values,
                )
            for i, si in enumerate(all_s):
                dfi = df.loc[(df[S_col] == si) & ind_xKnow, :]
                if dfi.shape[0] == 0:
                    continue
                pm.Normal(
                    "W_%s_obs_X" % str(si),
                    a_s[i] + b_s[i] * dfi[X_col].values,
                    sigma_ws[i],
                    observed=dfi[W_col].values,
                )
                pm.Bernoulli(
                    "Y_%s_obs_X" % str(si),
                    logit_p=beta0s[i] + betax * dfi[X_col].values,
                    observed=dfi[Y_col].values,
                )

            if self._solver != "vi":
                self._res = pm.sample(
                    self._nsample,
                    tune=self._ntunes,
                    chains=self._nchains,
                    progressbar=self._pbar,
                    random_seed=list(
                        range(self._seed, self._seed + self._nchains)
                    ),
                    nuts_sampler=self._solver,
                )
            elif self._solver == "vi":
                self._res = pm.fit(
                    progressbar=self._pbar, random_seed=self._seed
                )

    def summary(
        self,
        var_names: Optional[Tuple[str]] = ("a_s", "b_s", "betax"),
        return_obj: Literal["raw", "point_interval"] = "point_interval",
    ) -> Union[pd.DataFrame, az.InferenceData]:
        assert return_obj in ["raw", "point_interval"]
        if self._solver != "vi":
            if return_obj == "point_interval":
                res_df = az.summary(
                    self._res,
                    hdi_prob=0.95,
                    kind="stats",
                    var_names=list(var_names),
                )
        else:
            if return_obj == "point_interval":
                # TODO: 有一些param是log__之后的
                nparam = self._res.ndim
                param_names = [None] * nparam
                # log_slices = []
                if var_names is not None:
                    post_mu = self._res.mean.eval()
                    post_sd = self._res.std.eval()
                    res_df = []
                    for vari in var_names:
                        if vari in self._res.ordering:
                            pass
                        elif (vari + "_log__") in self._res.ordering:
                            vari = vari + "_log__"
                        else:
                            raise KeyError(
                                "%s or %s_log__ not in approximation.ordering."
                                % (vari, vari)
                            )
                        slice_ = self._res.ordering[vari][1]
                        post_mu_i = post_mu[slice_]
                        post_sd_i = post_sd[slice_]
                        n = len(post_mu_i)
                        res_df.append(
                            pd.DataFrame(
                                {"mean": post_mu_i, "sd": post_sd_i},
                                index=[vari]
                                if n == 1
                                else ["%s[%d]" % (vari, i) for i in range(n)],
                            )
                        )
                    res_df = pd.concat(res_df)
                else:
                    for param, (_, slice_, _, _) in self._res.ordering.items():
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
                            "mean": self._res.mean.eval(),
                            "sd": self._res.std.eval(),
                        },
                        index=param_names,
                    )
                res_df["hdi_2.5%"] = res_df["mean"] - 1.96 * res_df["sd"]
                res_df["hdi_97.5%"] = res_df["mean"] + 1.96 * res_df["sd"]

        if return_obj == "raw":
            return self._res
        elif return_obj == "point_interval":
            return res_df
