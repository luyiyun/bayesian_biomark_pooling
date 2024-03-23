import logging
import os
import multiprocessing as mp
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
import statsmodels.api as sm

from bayesian_biomarker_pooling.simulate import Simulator
from bayesian_biomarker_pooling.embp import EMBP


def plot_params_hist(params_hist: np.ndarray, names: np.ndarray, savefn: str):
    nparams = params_hist.shape[1]
    nr = int(np.sqrt(nparams))
    nc = (nparams + 1) // nr
    fig, axs = plt.subplots(
        nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), sharey=False
    )
    axs = axs.flatten()
    for i in range(nparams):
        pd.Series(params_hist[:, i]).plot(ax=axs[i])
        axs[i].set_title(names[i])
    fig.tight_layout()
    fig.savefig(savefn)


def temp_test_continue(ci=False, ve_method="bootstrap", seed=0):
    log_level = logging.WARNING
    logger = logging.getLogger("EMBP")
    logger.setLevel(log_level)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(log_level)

    root = "./results/tmp_embp"
    os.makedirs(root, exist_ok=True)

    simulator = Simulator(type_outcome="continue")
    df = simulator.simulate(seed)
    model = EMBP(
        outcome_type="continue",
        variance_estimate=ci,
        variance_estimate_method=ve_method,
        pbar=True,
        seed=seed,
    )
    model.fit(df["X"].values, df["S"].values, df["W"].values, df["Y"].values)

    params = model.params_
    params_hist = model.params_hist_
    plot_params_hist(
        params_hist.values,
        params_hist.columns.values,
        os.path.join(root, "params_hist.png"),
    )
    if ci and (ve_method == "sem"):
        R = model._estimator._R
        np.save("./temp_R.npy", R)
        plot_params_hist(
            R[:, 14, :],
            params_hist.columns.values,
            os.path.join(root, "R_beta_x.png"),
        )

    print(params)


def temp_test_binary(ci=False, seed=0):
    log_level = logging.WARNING
    logger = logging.getLogger("EMBP")
    logger.setLevel(log_level)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(log_level)

    root = "./results/tmp_embp"
    os.makedirs(root, exist_ok=True)

    simulator = Simulator(type_outcome="binary")
    df = simulator.simulate(seed)
    model = EMBP(
        outcome_type="binary",
        variance_estimate=ci,
        variance_estimate_method="bootstrap",
        max_iter=300,
        pbar=True,
        n_importance_sampling=1000,
        use_gpu=True,
        seed=seed,
    )
    model.fit(df["X"].values, df["S"].values, df["W"].values, df["Y"].values)

    params = model.params_
    params_hist = model.params_hist_
    model.params_.to_csv(os.path.join(root, "temp_params.csv"))
    params_hist.to_csv(os.path.join(root, "temp_hist.csv"))

    plot_params_hist(
        params_hist.values,
        params_hist.columns.values,
        os.path.join(root, "params_hist.png"),
    )

    print(params)


def method_xonly(df: pd.DataFrame) -> np.ndarray:
    df_drop = df.dropna()
    x = df_drop["X"].values
    y = df_drop["Y"].values
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    res = model.fit()
    return np.r_[res.params[-1], res.conf_int()[1, :]]


def method_naive(df: pd.DataFrame) -> np.ndarray:
    x = df["W"].values
    y = df["Y"].values
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    res = model.fit()
    return np.r_[res.params[-1], res.conf_int()[1, :]]


def trial_once_by_simulator_and_estimator(
    simulator: Simulator,
    estimator: EMBP,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    # 1. generate data
    df = simulator.simulate(seed=seed)

    # 2. other methods
    res_xonly = method_xonly(df)
    res_naive = method_naive(df)

    # 3. run model
    estimator.fit(
        df["X"].values, df["S"].values, df["W"].values, df["Y"].values
    )
    res_embp = estimator.params_.values

    return {"EMBP": res_embp, "naive": res_naive, "xonly": res_xonly}


def trial(
    root: str,
    seed: int = 0,
    repeat: int = 100,
    beta_x: float = 1.0,
    n_sample_per_studies: int = 100,
    n_knowX_per_studies: int = 10,
    ci: bool = False,
    ci_method: Literal["sem", "bootstrap"] = "sem",
    max_iter: int = 1000,
    n_cores: int = 1,
    type_outcome: Literal["continue", "binary"] = "continue",
    use_gpu: bool = True,
):
    log_level = logging.ERROR  # 将warning去掉
    logger = logging.getLogger("EMBP")
    logger.setLevel(log_level)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(log_level)

    # 模拟实验：
    # 1. 不同样本量，不同缺失比例下的效果,
    # 2. 一类错误 & 效能
    # 3. 把参数默认值搞清楚

    simulator = Simulator(
        type_outcome=type_outcome,
        beta_x=beta_x,
        sigma2_y=[0.5, 0.75, 1.0, 1.25],
        sigma2_e=[0.5, 0.75, 1.0, 1.25],
        beta_0=[-0.5, -0.25, 0.25, 0.5],
        n_sample_per_studies=n_sample_per_studies,
        n_knowX_per_studies=n_knowX_per_studies,
    )
    params_ser = simulator.parameters_series

    embp_model = EMBP(
        outcome_type=type_outcome,
        variance_estimate=ci,
        variance_estimate_method=ci_method,
        pbar=False,
        max_iter=max_iter,
        seed=seed,
    )

    res_arrs = {}
    if n_cores <= 1:
        for trial_i in tqdm(range(repeat)):
            resi = trial_once_by_simulator_and_estimator(
                simulator=simulator, estimator=embp_model, seed=trial_i + seed
            )
            for k, arr in resi.items():
                res_arrs.setdefault(k, []).append(arr)
    else:
        with mp.Pool(n_cores) as pool:
            tmp_reses = [
                pool.apply_async(
                    trial_once_by_simulator_and_estimator,
                    (
                        simulator,
                        embp_model,
                        trial_i + seed,
                    ),
                )
                for trial_i in range(repeat)
            ]
            for tmp_resi in tqdm(tmp_reses):
                resi = tmp_resi.get()
                for k, arr in resi.items():
                    res_arrs.setdefault(k, []).append(arr)

    # 3. collect results
    res_all = {
        "true": xr.DataArray(
            params_ser.values,
            dims=("params",),
            coords={"params": params_ser.index.values},
        ),
    }
    for k, arrs in res_arrs.items():
        if k == "EMBP":
            res_all[k] = xr.DataArray(
                np.stack(arrs, axis=0),
                dims=("repeat", "params", "statistic"),
                coords={
                    "params": params_ser.index.values,
                    "statistic": embp_model.result_columns,
                },
            )
        else:
            res_all[k] = xr.DataArray(
                np.stack(arrs, axis=0)[:, None, :],
                dims=("repeat", "params", "statistic"),
                coords={
                    "params": ["beta_x"],
                    "statistic": ["estimate", "CI_1", "CI_2"],
                },
            )
    res = xr.Dataset(
        res_all,
        attrs={
            "seed": seed,
            "repeat": repeat,
            "n_sample_per_studies": n_sample_per_studies,
            "n_knowX_per_studies": n_knowX_per_studies,
            "ci": ci,
            "ci_method": ci_method,
            "max_iter": max_iter,
            "n_cores": n_cores,
            "type_outcome": type_outcome,
            "use_gpu": use_gpu,
        },
    )

    # 4. print simple summary
    summary = {}
    for method in ["EMBP", "xonly", "naive"]:
        summ_i = {}
        resi = res[method]
        diff = (
            resi.sel(params="beta_x", statistic="estimate").values
            - params_ser["beta_x"]
        )
        summ_i["bias"] = diff.mean()
        summ_i["mse"] = (diff**2).mean()
        if ci:
            in_ci = (
                resi.sel(params="beta_x", statistic="CI_1").values
                <= params_ser["beta_x"]
            ) & (
                resi.sel(params="beta_x", statistic="CI_2").values
                >= params_ser["beta_x"]
            )
            summ_i["cov_rate"] = in_ci.mean()
        summary[method] = summ_i
    summary = pd.DataFrame.from_dict(summary)
    print(summary)
    summary = xr.DataArray(
        summary.values,
        dims=("metric", "method"),
        coords={
            "metric": summary.index.values,
            "method": summary.columns.values,
        },
    )
    res["summary"] = summary

    # 5. save results
    os.makedirs(root, exist_ok=True)
    res.to_netcdf(
        os.path.join(
            root, f"{type_outcome}-{datetime.now(): %Y-%m-%d_%H-%M-%S}.nc"
        )
    )


# def trial_binary(ci=False):
#     # 模拟实验：
#     # 1. 不同样本量，不同缺失比例下的效果,
#     # 2. 一类错误 & 效能
#     # 3. 把参数默认值搞清楚

#     simulator = Simulator(
#         type_outcome="binary",
#         beta_x=1.0,
#         sigma2_y=[0.5, 1.0, 1.25, 1.5],
#         beta_0=[-0.5, -0.25, 0.25, 0.5],
#     )
#     res_em, res_ols = [], []
#     # res_em_ci1, res_em_ci2 = [], []
#     for i in tqdm(range(100)):
#         df = simulator.simulate()
#         model = EMBP(
#             outcome_type="binary",
#             variance_estimate=False,
#             pbar=False,
#             max_iter=300,
#             n_importance_sampling=1000,
#             use_gpu=True,
#         )
#         model.fit(
#             df["X"].values, df["S"].values, df["W"].values, df["Y"].values
#         )
#         resi = model.params_.loc["beta_x", "estimate"]
#         res_em.append(resi)
#         # res_em_ci1.append(model.params_.loc["beta_x", "CI_1"])
#         # res_em_ci2.append(model.params_.loc["beta_x", "CI_2"])
#         res_ols.append(model.params_hist_["beta_x"].iloc[0])
#     true_beta_x = simulator.parameters["beta_x"]
#     print(f"True: {true_beta_x: .6f}")
#     res_em, res_ols = np.array(res_em), np.array(res_ols)
#     # res_ci1, res_ci2 = np.array(res_em_ci1), np.array(res_em_ci2)
#     # cov_rate = np.mean((res_ci1 <= true_beta_x) & (res_ci2 >= true_beta_x))
#     print(
#         f"OLS: {res_ols.mean(): .6f}, "
#         f"Bias is {np.abs(res_ols.mean() - true_beta_x):.6f},"
#         f" MSE is {np.mean((res_ols - true_beta_x) ** 2):.6f}"
#     )
#     print(
#         f"EMBP: {res_em.mean(): .6f}"
#         f", Bias is {np.abs(res_em.mean() - true_beta_x):.6f}"
#         f", MSE is {np.mean((res_em - true_beta_x) ** 2):.6f}, "
#         # f", Cov Rate is {cov_rate: .6f}"
#     )
#     # simulator = Simulator(type_outcome="binary", n_knowX_per_studies=10)
#     # df = simulator.simulate()
#     # print(df)
#     # model = EMBP(outcome_type="binary")
#     # model.fit(df["X"].values, df["S"].values,
#                 df["W"].values, df["Y"].values)
#     # print(simulator.parameters["beta_x"], model.params_["beta_x"])


def main():
    # temp_test_continue(ci=True, ve_method="bootstrap")
    # temp_test_binary(ci=True)
    # trial_binary(ci=False)
    for n_sample_per_studies in [50, 100, 150, 200]:
        for know_x_ratio in [0.1, 0.15, 0.2]:
            print(
                f"nSamplePerStudy={n_sample_per_studies},"
                f" RatioXKnow={know_x_ratio}"
            )
            n_know_x_per_studies = int(n_sample_per_studies * know_x_ratio)
            if n_sample_per_studies == 100 and n_know_x_per_studies == 10:
                continue
            trial(
                root="./results/embp",
                type_outcome="continue",
                repeat=1000,
                ci=True,
                ci_method="sem",
                n_cores=1,
                beta_x=1.0,
                n_sample_per_studies=n_sample_per_studies,
                n_knowX_per_studies=n_know_x_per_studies,
            )


if __name__ == "__main__":
    main()
