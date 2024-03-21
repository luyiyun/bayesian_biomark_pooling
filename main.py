import logging
import os

import numpy as np

# import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from bayesian_biomarker_pooling.simulate import Simulator
from bayesian_biomarker_pooling.embp import EMBP


def temp_test_continue():
    log_level = logging.INFO
    logger = logging.getLogger("EMBP")
    logger.setLevel(log_level)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(log_level)

    root = "./results/tmp_embp"
    os.makedirs(root, exist_ok=True)

    simulator = Simulator(type_outcome="continue")
    df = simulator.simulate()
    model = EMBP(
        outcome_type="continue",
        variance_estimate=False,
        pbar=True,
    )
    model.fit(df["X"].values, df["S"].values, df["W"].values, df["Y"].values)

    params = model.params_
    params_hist = model._estimator.params_hist_
    # R = model._estimator._R

    model.params_.to_csv(os.path.join(root, "temp_params.csv"))
    params_hist.to_csv(os.path.join(root, "temp_hist.csv"))
    # np.save("./temp_R.npy", R)

    # params = pd.read_csv("./temp_params.csv", index_col=0)
    # params_hist = pd.read_csv("./temp_hist.csv", index_col=0)
    # R = np.load("./temp_R.npy")
    print(params)

    nr = int(np.sqrt(params.shape[0]))
    nc = (params.shape[0] + 1) // nr
    fig, axs = plt.subplots(
        nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), sharey=False
    )
    axs = axs.flatten()
    for i in range(params_hist.shape[1]):
        params_hist.iloc[:, i].plot(ax=axs[i])
        axs[i].set_title(params.index[i])
    # df = pd.DataFrame(R[:, 14, :], columns=params_hist.columns)
    # df.plot(ax=axs[1])
    fig.tight_layout()
    fig.savefig(os.path.join(root, "params_hist.png"))
    # plt.show()


def temp_test_binary():
    log_level = logging.WARNING
    logger = logging.getLogger("EMBP")
    logger.setLevel(log_level)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(log_level)

    root = "./results/tmp_embp"
    os.makedirs(root, exist_ok=True)

    simulator = Simulator(type_outcome="binary")
    df = simulator.simulate()
    model = EMBP(
        outcome_type="binary",
        max_iter=500,
        variance_estimate=False,
        variance_esitmate_method="sem",
        thre=1e-3,
        thre_inner=1e-10,
        thre_var_est=1e-4,
        pbar=True,
        n_importance_sampling=10000,
        ema=0.1,
        use_gpu=True,
    )
    model.fit(df["X"].values, df["S"].values, df["W"].values, df["Y"].values)

    params = model.params_
    params_hist = model._estimator.params_hist_
    # R = model._estimator._R

    model.params_.to_csv(os.path.join(root, "temp_params.csv"))
    params_hist.to_csv(os.path.join(root, "temp_hist.csv"))
    # np.save("./temp_R.npy", R)

    # params = pd.read_csv("./temp_params.csv", index_col=0)
    # params_hist = pd.read_csv("./temp_hist.csv", index_col=0)
    # R = np.load("./temp_R.npy")
    print(params)

    nr = int(np.sqrt(params.shape[0]))
    nc = (params.shape[0] + 1) // nr
    fig, axs = plt.subplots(
        nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), sharey=False
    )
    axs = axs.flatten()
    for i in range(params_hist.shape[1]):
        params_hist.iloc[:, i].plot(ax=axs[i])
        axs[i].set_title(params.index[i])
    # df = pd.DataFrame(R[:, 14, :], columns=params_hist.columns)
    # df.plot(ax=axs[1])
    fig.tight_layout()
    fig.savefig(os.path.join(root, "params_hist.png"))
    # plt.show()


def trial_continue(ci=False):
    # 模拟实验：
    # 1. 不同样本量，不同缺失比例下的效果,
    # 2. 一类错误 & 效能
    # 3. 把参数默认值搞清楚

    simulator = Simulator(
        type_outcome="continue",
        beta_x=1.0,
        sigma2_y=[0.5, 1.0, 1.25, 1.5],
        beta_0=[0.5, 0.75, 1.25, 1.5],
    )
    res_em, res_ols = [], []
    if ci:
        res_em_ci1, res_em_ci2 = [], []
    for i in tqdm(range(100)):
        df = simulator.simulate()
        model = EMBP(
            outcome_type="continue",
            variance_estimate=False,
            pbar=False,
        )
        model.fit(
            df["X"].values, df["S"].values, df["W"].values, df["Y"].values
        )
        resi = model.params_.loc["beta_x", "estimate"]
        res_em.append(resi)
        res_ols.append(model._estimator.params_hist_["beta_x"].iloc[0])
        if ci:
            res_em_ci1.append(model.params_.loc["beta_x", "CI_1"])
            res_em_ci2.append(model.params_.loc["beta_x", "CI_2"])
    true_beta_x = simulator.parameters["beta_x"]
    print(f"True: {true_beta_x: .6f}")
    res_em, res_ols = np.array(res_em), np.array(res_ols)
    if ci:
        res_ci1, res_ci2 = np.array(res_em_ci1), np.array(res_em_ci2)
        cov_rate = np.mean((res_ci1 <= true_beta_x) & (res_ci2 >= true_beta_x))
    print(
        f"OLS: {res_ols.mean(): .6f}, "
        f"Bias is {np.abs(res_ols.mean() - true_beta_x):.6f},"
        f" MSE is {np.mean((res_ols - true_beta_x) ** 2):.6f}"
    )
    print(
        f"EMBP: {res_em.mean(): .6f}"
        f", Bias is {np.abs(res_em.mean() - true_beta_x):.6f}"
        f", MSE is {np.mean((res_em - true_beta_x) ** 2):.6f}, "
    )
    print(f", Cov Rate is {cov_rate: .6f}" if ci else "")
    # simulator = Simulator(type_outcome="binary", n_knowX_per_studies=10)
    # df = simulator.simulate()
    # print(df)
    # model = EMBP(outcome_type="binary")
    # model.fit(df["X"].values, df["S"].values, df["W"].values, df["Y"].values)
    # print(simulator.parameters["beta_x"], model.params_["beta_x"])


def trial_binary():
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
            outcome_type="binary", variance_estimate=False, pbar=False, ema=0.5
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
    # temp_test_continue()
    # trial_binary()
    trial_continue(ci=False)


if __name__ == "__main__":
    main()
