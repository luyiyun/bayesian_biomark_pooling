from typing import Literal
from copy import deepcopy

import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy import ndarray
from numpy.random import Generator

from ..base import BiomarkerPoolBase
from .base import EM
from .continuous import ContinueEM
# from .binary import BinaryEM
from .binary_lap import LapBinaryEM


def bootstrap_estimator(
    estimator: EM,
    X: ndarray,
    Y: ndarray,
    W: ndarray,
    S: ndarray,
    Z: ndarray | None = None,
    Y_type: Literal["continue", "binary"] = False,
    n_repeat: int = 200,
    seed: int | None | Generator = None,
    pbar: bool = True,
) -> pd.DataFrame:
    assert hasattr(
        estimator, "params_"
    ), "please run regular EM iteration firstly!"

    seed = np.random.default_rng(seed)
    ind_bootstrap = seed.choice(
        Y.shape[0], (n_repeat, Y.shape[0]), replace=True
    )

    estimator._pbar = False

    init_params = estimator.params_.copy()
    params_bs = []
    for i in tqdm(range(n_repeat), disable=not pbar):
        ind_bs = ind_bootstrap[i]
        estimator.register_data(
            X[ind_bs],
            S[ind_bs],
            W[ind_bs],
            Y[ind_bs],
            None if Z is None else Z[ind_bootstrap],  # nbs x N x nz
        )
        estimator.run(init_params=init_params)
        params_bs.append(estimator.params_)

    return np.stack(params_bs, axis=0)


class EMBP(BiomarkerPoolBase):
    def __init__(
        self,
        outcome_type: Literal["continue", "binary"],
        max_iter: int | None = None,
        max_iter_inner: int = 100,
        delta1: float = 1e-3,
        delta1_inner: float = 1e-4,
        delta2: float | None = None,
        delta2_inner: float = 1e-7,
        delta1_var: float = 1e-1,
        delta2_var: float = 1e-3,
        ci: bool = False,
        ci_method: Literal["sem", "bootstrap"] = "bootstrap",
        ci_level: float = 0.95,
        n_bootstrap: int = 200,
        pbar: bool = True,
        seed: int | None = 0,
        use_gpu: bool = False,
        quasi_mc_K: int = 1000,
        gem: bool = False,
    ) -> None:
        """
        delta2: 1e-5 for continue, 1e-2 for binary
        max_iter: 500 for continue, 300 for binary
        """
        assert outcome_type in ["continue", "binary"]
        assert ci_method in ["bootstrap"]
        if use_gpu:
            try:
                import torch
            except ImportError:
                raise ImportError(
                    "torch is not installed, "
                    "please install torch or BBP[torch]"
                )
        if use_gpu and outcome_type == "continue":
            raise NotImplementedError
        # if outcome_type == "binary" and ci:
        #     raise NotImplementedError
        # if outcome_type == "binary" and variance_estimate_method == "sem":
        #     raise NotImplementedError(
        #         "use bootstrap for outcome_type = binary"
        #     )

        self.outcome_type_ = outcome_type
        self.max_iter_ = (
            max_iter or {"continue": 500, "binary": 300}[outcome_type]
        )
        self.max_iter_inner_ = max_iter_inner
        self.delta1_ = delta1
        self.delta1_inner_ = delta1_inner
        self.delta2_ = (
            delta2 or {"continue": 1e-5, "binary": 1e-3}[outcome_type]
        )
        self.delta2_inner_ = delta2_inner
        self.delta1_var_ = delta1_var
        self.delta2_var_ = delta2_var
        # self.min_nIS_ = min_nIS
        # self.max_nIS_ = max_nIS
        # self.lr_ = lr
        self.pbar_ = pbar
        self.ci_ = ci
        self.ci_method_ = ci_method
        self.ci_level_ = ci_level
        self.n_bootstrap_ = n_bootstrap
        self.use_gpu_ = use_gpu
        self.seed_ = np.random.default_rng(seed)
        self.gem_ = gem

        self.quasi_mc_K_ = quasi_mc_K

        if use_gpu and seed is not None:
            torch.random.manual_seed(seed)

    @property
    def result_columns(self):
        if not self.ci_:
            return ["estimate"]
        if self.ci_method_ == "bootstrap":
            return ["estimate", "CI_1", "CI_2"]
        return ["estimate", "variance(log)", "std(log)", "CI_1", "CI_2"]

    @property
    def result_index(self):
        return self._estimator.get_params_names()

    def fit(
        self,
        X: ndarray,
        S: ndarray,
        W: ndarray,
        Y: ndarray,
        Z: ndarray | None = None,
    ) -> None:
        if self.outcome_type_ == "continue":
            self._estimator = ContinueEM(
                max_iter=self.max_iter_,
                max_iter_inner=self.max_iter_inner_,
                delta1=self.delta1_,
                delta1_inner=self.delta1_inner_,
                delta2=self.delta2_,
                delta2_inner=self.delta2_inner_,
                delta1_var=self.delta1_var_,
                delta2_var=self.delta2_var_,
                pbar=self.pbar_,
            )
        elif self.outcome_type_ == "binary":
            if self.use_gpu_:
                from .binary_gpu import BinaryEMTorch

                self._estimator = BinaryEMTorch(
                    max_iter=self.max_iter_,
                    max_iter_inner=self.max_iter_inner_,
                    delta1=self.delta1_,
                    delta1_inner=self.delta1_inner_,
                    delta2=self.delta2_,
                    delta2_inner=self.delta2_inner_,
                    delta1_var=self.delta1_var_,
                    delta2_var=self.delta2_var_,
                    pbar=self.pbar_,
                    device="cuda:0",
                )
            else:
                self._estimator = LapBinaryEM(
                    max_iter=self.max_iter_,
                    max_iter_inner=self.max_iter_inner_,
                    delta1=self.delta1_,
                    delta1_inner=self.delta1_inner_,
                    delta1_var=self.delta1_var_,
                    delta2=self.delta2_,
                    delta2_inner=self.delta2_inner_,
                    delta2_var=self.delta2_var_,
                    pbar=self.pbar_,
                    random_seed=self.seed_,
                    K=self.quasi_mc_K_,
                    gem=self.gem_
                )
        self._estimator.register_data(X, S, W, Y, Z)
        self._estimator.run()

        params_names = []
        for k, v in self._estimator._params_ind.items():
            if isinstance(v, slice):
                params_names.extend([k] * (v.stop - v.start))
            else:
                params_names.append(k)
        self.params_ = pd.DataFrame(
            {"estimate": self._estimator.params_}, index=params_names
        )
        self.params_hist_ = pd.DataFrame(
            self._estimator.params_hist_, columns=params_names
        )

        if not self.ci_:
            return

        quan1 = (1 - self.ci_level_) / 2
        quan2 = 1 - quan1

        if self.ci_method_ == "bootstrap":
            # 使用boostrap方法
            if self.pbar_:
                print("Bootstrap: ")
            res_bootstrap = bootstrap_estimator(
                # 使用复制品，而非原始的estimator
                estimator=deepcopy(self._estimator),
                X=X,
                Y=Y,
                W=W,
                S=S,
                Z=Z,
                Y_type=self.outcome_type_,
                n_repeat=self.n_bootstrap_,
                seed=self.seed_,
                pbar=self.pbar_,
            )
            res_ci = np.quantile(
                res_bootstrap,
                q=[quan1, quan2],
                axis=0,
            )
            self.params_["CI_1"] = res_ci[0, :]
            self.params_["CI_2"] = res_ci[1, :]
        else:
            raise NotImplementedError("没有考虑ci_level")
            # params_var_ = self._estimator.estimate_variance()
            # self.params_["variance(log)"] = params_var_
            # self.params_["std(log)"] = np.sqrt(params_var_)
            # self.params_["CI_1"] = (
            #     self.params_["estimate"] - 1.96 * self.params_["std(log)"]
            # )
            # self.params_["CI_2"] = (
            #     self.params_["estimate"] + 1.96 * self.params_["std(log)"]
            # )
            # is_sigma2 = self.params_.index.map(
            #     lambda x: x.startswith("sigma2")
            # )
            # self.params_.loc[is_sigma2, "CI_1"] = np.exp(
            #     np.log(self.params_.loc[is_sigma2, "estimate"])
            #     - 1.96 * self.params_.loc[is_sigma2, "std(log)"]
            # )
            # self.params_.loc[is_sigma2, "CI_2"] = np.exp(
            #     np.log(self.params_.loc[is_sigma2, "estimate"])
            #     + 1.96 * self.params_.loc[is_sigma2, "std(log)"]
            # )
