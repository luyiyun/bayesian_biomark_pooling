import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm

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


def trial_continue(
    root: str,
    seed: int = 0,
    repeat: int = 100,
    beta_x: float = 1.0,
    n_sample_per_studies: int = 100,
    n_knowX_per_studies: int = 10,
    ci: bool = False,
    ci_method: str = "sem",
    max_iter: int = 1000,
):
    # 模拟实验：
    # 1. 不同样本量，不同缺失比例下的效果,
    # 2. 一类错误 & 效能
    # 3. 把参数默认值搞清楚

    simulator = Simulator(
        type_outcome="continue",
        beta_x=beta_x,
        sigma2_y=[0.5, 0.75, 1.0, 1.25],
        sigma2_e=[0.5, 0.75, 1.0, 1.25],
        beta_0=[-0.5, -0.25, 0.25, 0.5],
        n_sample_per_studies=n_sample_per_studies,
        n_knowX_per_studies=n_knowX_per_studies,
    )
    params_ser = simulator.parameters_series

    res_array = []
    for trial_i in tqdm(range(repeat)):
        # 1. generate data
        df = simulator.simulate(seed=trial_i + seed)

        # 2. run model
        model = EMBP(
            outcome_type="continue",
            variance_estimate=ci,
            variance_estimate_method=ci_method,
            pbar=False,
            max_iter=max_iter,
        )
        model.fit(
            df["X"].values, df["S"].values, df["W"].values, df["Y"].values
        )
        res_array.append(model.params_.values)

        # 3. other methods


    # 3. collect results
    res_array = np.stack(res_array, axis=0)
    res_array = xr.DataArray(
        res_array,
        dims=("repeat", "params", "statistic"),
        coords={
            "params": params_ser.index.values,
            "statistic": model.params_.columns.values,
        },
    )
    res = xr.Dataset(
        {
            "EMBP": res_array,
            "true": xr.DataArray(
                params_ser.values,
                dims=("params",),
                coords={"params": params_ser.index.values},
            ),
        },
        attrs={
            "n_sample_per_studies": n_sample_per_studies,
            "n_knowX_per_studies": n_knowX_per_studies,
        },
    )

    # 4. print simple summary
    summary = {}
    for method in ["EMBP"]:
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
        os.path.join(root, f"continue-{datetime.now(): %Y-%m-%d_%H-%M-%S}.nc")
    )


def trial_binary(ci=False):
    # 模拟实验：
    # 1. 不同样本量，不同缺失比例下的效果,
    # 2. 一类错误 & 效能
    # 3. 把参数默认值搞清楚

    simulator = Simulator(
        type_outcome="binary",
        beta_x=1.0,
        sigma2_y=[0.5, 1.0, 1.25, 1.5],
        beta_0=[-0.5, -0.25, 0.25, 0.5],
    )
    res_em, res_ols = [], []
    # res_em_ci1, res_em_ci2 = [], []
    for i in tqdm(range(100)):
        df = simulator.simulate()
        model = EMBP(
            outcome_type="binary",
            variance_estimate=False,
            pbar=False,
            max_iter=300,
            n_importance_sampling=1000,
            use_gpu=True,
        )
        model.fit(
            df["X"].values, df["S"].values, df["W"].values, df["Y"].values
        )
        resi = model.params_.loc["beta_x", "estimate"]
        res_em.append(resi)
        # res_em_ci1.append(model.params_.loc["beta_x", "CI_1"])
        # res_em_ci2.append(model.params_.loc["beta_x", "CI_2"])
        res_ols.append(model.params_hist_["beta_x"].iloc[0])
    true_beta_x = simulator.parameters["beta_x"]
    print(f"True: {true_beta_x: .6f}")
    res_em, res_ols = np.array(res_em), np.array(res_ols)
    # res_ci1, res_ci2 = np.array(res_em_ci1), np.array(res_em_ci2)
    # cov_rate = np.mean((res_ci1 <= true_beta_x) & (res_ci2 >= true_beta_x))
    print(
        f"OLS: {res_ols.mean(): .6f}, "
        f"Bias is {np.abs(res_ols.mean() - true_beta_x):.6f},"
        f" MSE is {np.mean((res_ols - true_beta_x) ** 2):.6f}"
    )
    print(
        f"EMBP: {res_em.mean(): .6f}"
        f", Bias is {np.abs(res_em.mean() - true_beta_x):.6f}"
        f", MSE is {np.mean((res_em - true_beta_x) ** 2):.6f}, "
        # f", Cov Rate is {cov_rate: .6f}"
    )
    # simulator = Simulator(type_outcome="binary", n_knowX_per_studies=10)
    # df = simulator.simulate()
    # print(df)
    # model = EMBP(outcome_type="binary")
    # model.fit(df["X"].values, df["S"].values, df["W"].values, df["Y"].values)
    # print(simulator.parameters["beta_x"], model.params_["beta_x"])


def main():
    # temp_test_continue(ci=True, ve_method="bootstrap")
    # temp_test_binary(ci=True)
    # trial_binary(ci=False)
    trial_continue(root="./results/embp", repeat=10, ci=True)


if __name__ == "__main__":
    main()
