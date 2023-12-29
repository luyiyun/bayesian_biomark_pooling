from typing import Literal, Tuple, Union, List
import numpy as np
import pymc as pm

from ._base import Model


class PymcModel(Model):
    def _solve_model(self):
        with self._model:
            if self._solver != "vi":
                res = pm.sample(
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
                res = pm.fit(progressbar=self._pbar, random_seed=self._seed)
        return res


class ModelwLikelihood(PymcModel):
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
        hier_prior_on_x: bool = True,
    ) -> None:
        super().__init__(nsample, ntunes, nchains, pbar, solver, seed)
        self._hier_prior_on_x = hier_prior_on_x

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
                a_s,
                b_s,
                sigma_ws,
                beta0s,
                betax,
            ) = self._set_prior(n_studies, n_xKnow, n_xUnKnow, X_Know)

            if self._hier_prior_on_x:
                mu_x = pm.Normal("mu_x", 0, 10)

                m, sigma = 1, 1
                beta = (m + np.sqrt(m + sigma**2)) / (2 * sigma**2)
                alpha = m * beta + 1
                sigma_x = pm.Gamma("sigma_x", alpha=alpha, beta=beta)
            else:
                mu_x, sigma_x = X_Know.mean(), X_Know.std()

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
        hier_prior_on_x: bool = True,
        prior_betax: Literal["flat", "normal"] = "flat",
    ) -> None:
        assert prior_betax in ["flat", "normal"]
        super().__init__(
            nsample, ntunes, nchains, pbar, solver, seed, hier_prior_on_x
        )
        self._prior_betax = prior_betax

    def _set_prior(
        self, n_studies: int, n_xKnow: int, n_xUnKnow: int, X_Know: np.ndarray
    ) -> Tuple[pm.Distribution | np.ndarray]:
        if self._prior_betax == "flat":
            betax = pm.Flat("betax")
        elif self._prior_betax == "normal":
            betax = pm.Normal("betax", mu=0, sigma=1)

        a_s = pm.Flat("a_s", size=n_studies)
        b_s = pm.Flat("b_s", size=n_studies)
        sigma_ws = pm.HalfCauchy("sigma_ws", 1.0)

        beta0 = pm.Flat("beta0")

        return a_s, b_s, sigma_ws, beta0, betax


class HierachicalModel(SimpleModel):
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
        hier_prior_on_x: bool = True,
        prior_betax: Literal["flat", "normal"] = "flat",
        prior_sigma_ws: Literal["gamma", "inv_gamma"] = "gamma",
        prior_sigma_ab0: Literal["half_cauchy", "half_flat"] = "half_cauchy",
        prior_a_std: float = 10.,
        prior_b_std: float = 10.,
        prior_beta0_std: float = 10.,
    ) -> None:
        assert prior_sigma_ws in ["gamma", "inv_gamma"]
        assert prior_sigma_ab0 in ["half_cauchy", "half_flat"]
        super().__init__(
            nsample,
            ntunes,
            nchains,
            pbar,
            solver,
            seed,
            hier_prior_on_x,
            prior_betax,
        )
        self._prior_sigma_ws = prior_sigma_ws
        self._prior_sigma_ab0 = prior_sigma_ab0
        self._prior_a_std = prior_a_std
        self._prior_b_std = prior_b_std
        self._prior_beta0_std = prior_beta0_std

    def _set_prior(
        self, n_studies: int, n_xKnow: int, n_xUnKnow: int, X_Know: np.ndarray
    ) -> Tuple[pm.Distribution | np.ndarray]:
        if self._prior_betax == "flat":
            betax = pm.Flat("betax")
        elif self._prior_betax == "normal":
            betax = pm.Normal("betax", mu=0, sigma=1)

        mu_sigma_w = pm.HalfFlat("mu_sigma_w")  # mu_sigma_w这里其实是mode
        sigma_sigma_w = pm.HalfCauchy("sigma_sigma_w", 1.0)
        if self._prior_sigma_ws == "gamma":
            # beta = (
            #     mu_sigma_w + pm.math.sqrt(mu_sigma_w + sigma_sigma_w**2)
            # ) / (2 * pm.math.sqr(sigma_sigma_w))
            # alpha = mu_sigma_w * beta + 1
            sigma_ws = pm.Gamma(
                "sigma_ws",
                # alpha=alpha,
                # beta=beta,
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

        # TODO: a、b和beta0的先验分布可以设置的更加informative一些
        a = pm.Normal("a", 0, self._prior_a_std)
        b = pm.Normal("b", 0, self._prior_b_std)
        beta0 = pm.Normal("beta0", 0, self._prior_beta0_std)
        # beta0 = pm.Flat("beta0")
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

        return a_s, b_s, sigma_ws, beta0s, betax
