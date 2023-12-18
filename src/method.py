from typing import Literal, Optional, Tuple, Union, List
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


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
        seed: int = 0,
        prior_sigma_ws: Literal["gamma", "inv_gamma"] = "gamma",
        prior_sigma_ab0: Literal["half_cauchy", "half_flat"] = "half_cauchy",
        hier_prior_on_x: bool = True,
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

    def _create_model(
        self,
        n_studies: int,
        n_xKnow: int,
        n_xUnKnow: int,
        X_Know: np.ndarray,
        XWY_xKnow: List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]],
        WY_xUnKnow: List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    ) -> pm.Model:
        raise NotImplementedError

    def fit(
        self,
        df: pd.DataFrame,
        X_col: str = "X",
        S_col: str = "S",
        W_col: str = "W",
        Y_col: str = "Y",
    ) -> None:
        if not self._hier_prior_on_x:
            self._mu_x, self._sigma_x = np.mean(df[X_col]), np.std(df[X_col])

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

        self._model = self._create_model(
            n_studies, n_xKnow, n_xUnKnow, X_Know, XWY_xKnow, WY_xUnKnow
        )

        with self._model:
            if self._solver != "vi":
                self._res = pm.sample(
                    self._nsample,
                    tune=self._ntunes,
                    chains=self._nchains,
                    cores=self._nchains,
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


class ModelwLikelihood(Model):
    def _set_prior(
        self,
        n_studies: int,
        n_xKnow: int,
        n_xUnKnow: int,
        X_Know: np.ndarray,
    ) -> Tuple[Union[pm.Distribution, np.ndarray]]:
        raise NotImplementedError

    def _create_model(
        self,
        n_studies: int,
        n_xKnow: int,
        n_xUnKnow: int,
        X_Know: np.ndarray,
        XWY_xKnow: List[Tuple[np.ndarray, np.ndarray, np.ndarray] | None],
        WY_xUnKnow: List[Tuple[np.ndarray, np.ndarray, np.ndarray] | None],
    ) -> Model:
        with pm.Model() as model:
            (
                mu_x,
                sigma_x,
                a_s,
                b_s,
                sigma_ws,
                beta0s,
                betax,
            ) = self._set_prior(n_studies, n_xKnow, n_xUnKnow, X_Know)

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
            if self._hier_prior_on_x:
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


class SimpleModel(ModelwLikelihood):
    def _set_prior(
        self, n_studies: int, n_xKnow: int, n_xUnKnow: int, X_Know: np.ndarray
    ) -> Tuple[pm.Distribution | np.ndarray]:
        betax = pm.Flat("betax")

        if self._hier_prior_on_x:
            mu_x = pm.Normal("mu_x", 0, 10)
            sigma_x = pm.Gamma("sigma_x", mu=1, sigma=10)
        else:
            mu_x, sigma_x = X_Know.mean(), X_Know.std()

        a_s = pm.Flat("a_s", size=n_studies)
        b_s = pm.Flat("b_s", size=n_studies)
        sigma_ws = pm.HalfCauchy("sigma_ws", 1.0)

        beta0 = pm.Flat("beta0")

        return mu_x, sigma_x, a_s, b_s, sigma_ws, beta0, betax


class HierachicalModel(ModelwLikelihood):
    def _set_prior(
        self, n_studies: int, n_xKnow: int, n_xUnKnow: int, X_Know: np.ndarray
    ) -> Tuple[pm.Distribution | np.ndarray]:
        betax = pm.Flat("betax")

        if self._hier_prior_on_x:
            mu_x = pm.Normal("mu_x", 0, 10)
            sigma_x = pm.HalfCauchy("sigma_x", 1.0)
        else:
            mu_x, sigma_x = X_Know.mean(), X_Know.std()

        mu_sigma_w = pm.HalfFlat("mu_sigma_w")
        sigma_sigma_w = pm.HalfCauchy("sigma_sigma_w", 1.0)
        if self._prior_sigma_ws == "gamma":
            sigma_ws = pm.Gamma(
                "sigma_ws",
                mu=mu_sigma_w,
                sigma=sigma_sigma_w,
                size=n_studies,
            )
        elif self._prior_sigma_ws == "inv_gamma":
            sigma2_ws = pm.InverseGamma(
                "sigma2_ws",
                mu=mu_sigma_w,
                sigma=sigma_sigma_w,
                size=n_studies,
            )
            sigma_ws = pm.Deterministic("sigma_ws", pm.math.sqrt(sigma2_ws))

        a = pm.Normal("a", 0, 10)
        b = pm.Normal("b", 0, 10)
        beta0 = pm.Flat("beta0")
        if self._prior_sigma_ab0 == "half_cauchy":
            sigma_a = pm.HalfCauchy("sigma_a", 1.0)
            sigma_b = pm.HalfCauchy("sigma_b", 1.0)
            sigma_0 = pm.HalfCauchy("sigma_0", 1.0)
        elif self._prior_sigma_ab0 == "half_flat":
            sigma_a = pm.HalfFlat("sigma_a")
            sigma_b = pm.HalfFlat("sigma_b")
            sigma_0 = pm.HalfFlat("sigma_0")
        a_s = pm.Normal("a_s", a, sigma_a, size=n_studies)
        b_s = pm.Normal("b_s", b, sigma_b, size=n_studies)
        beta0s = pm.Normal("beta0s", beta0, sigma_0, size=n_studies)

        return mu_x, sigma_x, a_s, b_s, sigma_ws, beta0s, betax
