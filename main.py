import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from bayesian_biomarker_pooling.simulate import Simulator
from bayesian_biomarker_pooling.embp import EMBP


def temp_test():
    logger = logging.getLogger("EMBP")
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.DEBUG)

    simulator = Simulator(type_outcome="binary")
    df = simulator.simulate()
    model = EMBP(
        outcome_type="binary",
        max_iter=500,
        variance_estimate=False,
        variance_esitmate_method="sem",
        thre=1e-4,
        thre_inner=1e-10,
        thre_var_est=1e-4,
        pbar=True,
        nsample_IS=5000,
        ema=0.1
    )
    model.fit(
        df["X"].values, df["S"].values, df["W"].values, df["Y"].values
    )

    params_hist = model._estimator.params_hist_
    # R = model._estimator._R

    model.params_.to_csv("./temp_params.csv")
    params_hist.to_csv("./temp_hist.csv")
    # np.save("./temp_R.npy", R)

    params = pd.read_csv("./temp_params.csv", index_col=0)
    params_hist = pd.read_csv("./temp_hist.csv", index_col=0)
    # R = np.load("./temp_R.npy")
    print(params)

    fig, axs = plt.subplots(ncols=2, figsize=(8, 4), sharey=False)
    params_hist["beta_x"].plot(ax=axs[0])
    # df = pd.DataFrame(R[:, 14, :], columns=params_hist.columns)
    # df.plot(ax=axs[1])
    fig.tight_layout()
    plt.show()


def trial():
    # 模拟实验：
    # 1. 不同样本量，不同缺失比例下的效果,
    # 2. 一类错误 & 效能
    # 3. 把参数默认值搞清楚

    simulator = Simulator(
        type_outcome="continue",
        beta_x=0.,
        sigma2_y=[0.5, 1.0, 1.25, 1.5],
        beta_0=[0.5, 0.75, 1.25, 1.5],
    )
    res_em, res_em_ci1, res_em_ci2, res_ols = [], [], [], []
    for i in tqdm(range(100)):
        df = simulator.simulate()
        model = EMBP(
            outcome_type="continue",
            max_iter=1000,
            variance_estimate=True,
            variance_esitmate_method="sem",
            thre=1e-10,
            thre_inner=1e-10,
            pbar=False,
        )
        model.fit(
            df["X"].values, df["S"].values, df["W"].values, df["Y"].values
        )
        resi = model.params_.loc["beta_x", "estimate"]
        res_em.append(resi)
        res_em_ci1.append(model.params_.loc["beta_x", "CI_1"])
        res_em_ci2.append(model.params_.loc["beta_x", "CI_2"])
        res_ols.append(model._estimator.params_hist_ori_["beta_x"].iloc[0])
    true_beta_x = simulator.parameters["beta_x"]
    print(f"True: {true_beta_x: .6f}")
    res_em, res_ols = np.array(res_em), np.array(res_ols)
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
        f", Cov Rate is {cov_rate: .6f}"
    )
    # simulator = Simulator(type_outcome="binary", n_knowX_per_studies=10)
    # df = simulator.simulate()
    # print(df)
    # model = EMBP(outcome_type="binary")
    # model.fit(df["X"].values, df["S"].values, df["W"].values, df["Y"].values)
    # print(simulator.parameters["beta_x"], model.params_["beta_x"])


def main():
    # trial()
    temp_test()


if __name__ == "__main__":
    main()
