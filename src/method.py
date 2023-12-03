from typing import Literal, Optional, Tuple, Union

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


def bayesian_analysis(
    df: pd.DataFrame,
    nsample: int = 1000,
    ntunes: int = 1000,
    nchains: int = 1,
    pbar: bool = False,
    solver: Literal["pymc", "blackjax", "numpyro", "vi"] = "pymc",
    return_obj: Literal["raw", "point_interval"] = "point_interval",
    var_names: Optional[Tuple[str]] = ("a_s", "b_s", "betax"),
    seed: Optional[int] = None,
) -> Union[pd.DataFrame, az.InferenceData, pm.Approximation]:
    assert solver in ["pymc", "blackjax", "numpyro", "vi"]
    assert return_obj in ["raw", "point_interval"]

    mu_x, sigma_x = np.mean(df["X"]), np.std(df["X"])
    all_s = df["S"].unique()
    ns = all_s.shape[0]
    # N_X_no_obs = df["X"].isna().sum()

    with pm.Model():
        # a = pm.Flat("a")
        # b = pm.Flat("b")
        a = pm.Normal("a", 0, 10)
        b = pm.Normal("b", 0, 10)

        beta0 = pm.Flat("beta0")
        sigma_a = pm.HalfCauchy("sigma_a", 1.0)
        sigma_b = pm.HalfCauchy("sigma_b", 1.0)
        sigma_0 = pm.HalfCauchy("sigma_0", 1.0)
        alpha_sigma_w = pm.HalfCauchy("alpha_sigma_w", 1.0)
        beta_sigma_w = pm.HalfCauchy("beta_sigma_w", 1.0)
        # sigma_a = pm.HalfFlat("sigma_a")
        # sigma_b = pm.HalfFlat("sigma_b")
        # sigma_0 = pm.HalfFlat("sigma_0")
        # alpha_sigma_w = pm.HalfFlat("alpha_sigma_w")
        # beta_sigma_w = pm.HalfFlat("beta_sigma_w")

        betax = pm.Flat("betax")

        a_s = pm.Normal("a_s", a, sigma_a, size=ns)
        b_s = pm.Normal("b_s", b, sigma_b, size=ns)
        beta0s = pm.Normal("beta0s", beta0, sigma_0, size=ns)
        sigma_ws = pm.Gamma(
            "sigma_ws", alpha=alpha_sigma_w, beta=beta_sigma_w, size=ns
        )
        # X_no_obs = pm.Normal("X_no_obs", m_x, sigma_x, size=(N_X_no_obs,))

        # for samples that can not see X
        for i, si in enumerate(all_s):
            dfi = df.loc[(df["S"] == si) & df["X"].isna(), :]
            if dfi.shape[0] == 0:
                continue
            X_no_obs_i = pm.Normal(
                "X_no_obs_%s" % str(si), mu_x, sigma_x, size=(dfi.shape[0],)
            )
            pm.Normal(
                "W_%s_no_obs_X" % str(si),
                a_s[i] + b_s[i] * X_no_obs_i,
                sigma_ws[i],
                observed=dfi["W"].values,
            )
            pm.Bernoulli(
                "Y_%s_no_obs_X" % str(si),
                logit_p=beta0s[i] + betax * X_no_obs_i,
                observed=dfi["Y"].values,
            )

        # for samples that can see X
        for i, si in enumerate(all_s):
            dfi = df.loc[(df["S"] == si) & df["X"].notna(), :]
            if dfi.shape[0] == 0:
                continue
            pm.Normal(
                "W_%s_obs_X" % str(si),
                a_s[i] + b_s[i] * dfi["X"].values,
                sigma_ws[i],
                observed=dfi["W"].values,
            )
            pm.Bernoulli(
                "Y_%s_obs_X" % str(si),
                logit_p=beta0s[i] + betax * dfi["X"].values,
                observed=dfi["Y"].values,
            )

        if solver != "vi":
            res = pm.sample(
                nsample,
                tune=ntunes,
                chains=nchains,
                progressbar=pbar,
                random_seed=list(range(seed, seed + nchains)),
                nuts_sampler=solver,
            )
            if return_obj == "point_interval":
                res_df = az.summary(
                    res, hdi_prob=0.95, kind="stats", var_names=list(var_names)
                )
        elif solver == "vi":
            res = pm.fit(progressbar=pbar, random_seed=seed)
            if return_obj == "point_interval":
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
                                index=[vari]
                                if n == 1
                                else ["%s[%d]" % (vari, i) for i in range(n)],
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
                        {"mean": res.mean.eval(), "sd": res.std.eval()},
                        index=param_names,
                    )
                res_df["hdi_2.5%"] = res_df["mean"] - 1.96 * res_df["sd"]
                res_df["hdi_97.5%"] = res_df["mean"] + 1.96 * res_df["sd"]
    if return_obj == "raw":
        return res
    elif return_obj == "point_interval":
        return res_df
