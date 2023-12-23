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

    def _solve_model(self):
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
        self._res = self._solve_model()

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
