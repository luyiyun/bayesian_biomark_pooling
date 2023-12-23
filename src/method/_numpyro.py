# from typing import Tuple, Union, List

# from numpyro.infer import MCMC, NUTS
# from jax import random
# import numpyro.distributions as dist
# import arviz as az
# import numpy as np
# import numpyro as npro

# from ._base import Model


# class NumpyroModel(Model):
#     def _solve_model(self):
#         nuts_kernel = NUTS(self._model)
#         mcmc = MCMC(
#             nuts_kernel,
#             num_samples=self._nsample,
#             num_warmup=self._ntunes,
#             num_chains=self._nchains,
#             progress_bar=self._pbar,
#         )
#         mcmc.run(random.PRNGKey(self._seed))
#         return az.from_numpyro(mcmc)


# class NumpyroHierachicalModel(NumpyroModel):
#     def _create_model(
#         self,
#         n_studies: int,
#         n_xKnow: int,
#         n_xUnKnow: int,
#         X_Know: np.ndarray,
#         XWY_xKnow: List[Tuple[np.ndarray, np.ndarray, np.ndarray] | None],
#         WY_xUnKnow: List[Tuple[np.ndarray, np.ndarray, np.ndarray] | None],
#     ) -> Model:

#         def _model():
#             # prior
#             betax = npro.sample("betax", dist.Normal(0, 10))
#             mu_sigma_w = npro.sample("mu_sigma_w", dist.HalfNormal(10))

#         with pm.Model() as model:
#             (
#                 a_s,
#                 b_s,
#                 sigma_ws,
#                 beta0s,
#                 betax,
#             ) = self._set_prior(n_studies, n_xKnow, n_xUnKnow, X_Know)

#             if self._hier_prior_on_x:
#                 mu_x = pm.Normal("mu_x", 0, 10)

#                 m, sigma = 1, 1
#                 beta = (m + np.sqrt(m + sigma**2)) / (2 * sigma**2)
#                 alpha = m * beta + 1
#                 sigma_x = pm.Gamma("sigma_x", alpha=alpha, beta=beta)
#             else:
#                 mu_x, sigma_x = X_Know.mean(), X_Know.std()

#             if beta0s.ndim == 0:
#                 beta0s = [beta0s] * n_studies
#             if sigma_ws.ndim == 0:
#                 sigma_ws = [sigma_ws] * n_studies

#             # for samples that can not see X
#             X_no_obs = pm.Normal(
#                 "X_no_obs",
#                 mu_x,
#                 sigma_x,
#                 size=n_xUnKnow,
#             )
#             start = 0
#             for i, WY in enumerate(WY_xUnKnow):
#                 if WY is None:
#                     continue
#                 W, Y = WY
#                 end = start + W.shape[0]
#                 X_no_obs_i = X_no_obs[start:end]
#                 pm.Normal(
#                     "W_%d_no_obs_X" % i,
#                     a_s[i] + b_s[i] * X_no_obs_i,
#                     sigma_ws[i],
#                     observed=W,
#                 )
#                 pm.Bernoulli(
#                     "Y_%d_no_obs_X" % i,
#                     logit_p=beta0s[i] + betax * X_no_obs_i,
#                     observed=Y,
#                 )
#                 start = end

#             # for samples that can see X
#             if self._hier_prior_on_x:
#                 pm.Normal("X_obs", mu_x, sigma_x, observed=X_Know)
#             for i, XWY in enumerate(XWY_xKnow):
#                 if XWY is None:
#                     continue
#                 X, W, Y = XWY
#                 pm.Normal(
#                     "W_%d_obs_X" % i,
#                     a_s[i] + b_s[i] * X,
#                     sigma_ws[i],
#                     observed=W,
#                 )
#                 pm.Bernoulli(
#                     "Y_%d_obs_X" % i,
#                     logit_p=beta0s[i] + betax * X,
#                     observed=Y,
#                 )

#         return model
