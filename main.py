import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from bayesian_biomarker_pooling.simulate import Simulator
from bayesian_biomarker_pooling.embp import EMBP


def trial():
    simulator = Simulator(type_outcome="continue")
    res_em, res_ols = [], []
    for i in tqdm(range(1000)):
        df = simulator.simulate()
        model = EMBP(
            outcome_type="continue",
            max_iter=1000,
            variance_estimate=False,
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
        res_ols.append(model._estimator.params_hist_["beta_x"].iloc[0])
    true_beta_x = simulator.parameters["beta_x"]
    print(f"True: {true_beta_x: .6f}")
    res_em, res_ols = np.array(res_em), np.array(res_ols)
    print(
        f"OLS: {res_ols.mean(): .6f}, "
        f"Bias is {np.abs(res_ols.mean() - true_beta_x):.6f},"
        f" MSE is {np.mean((res_ols - true_beta_x) ** 2):.6f}"
    )
    print(
        f"EMBP: {res_em.mean(): .6f}, "
        f"Bias is {np.abs(res_em.mean() - true_beta_x):.6f},"
        f" MSE is {np.mean((res_em - true_beta_x) ** 2):.6f}"
    )
    # simulator = Simulator(type_outcome="binary", n_knowX_per_studies=10)
    # df = simulator.simulate()
    # print(df)
    # model = EMBP(outcome_type="binary")
    # model.fit(df["X"].values, df["S"].values, df["W"].values, df["Y"].values)
    # print(simulator.parameters["beta_x"], model.params_["beta_x"])


def main():
    logger = logging.getLogger("EMBP")
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.DEBUG)

    if not os.path.exists("./temp_params.csv"):
        simulator = Simulator(type_outcome="continue")
        df = simulator.simulate()
        model = EMBP(
            outcome_type="continue",
            max_iter=1000,
            variance_estimate=True,
            variance_esitmate_method="sem",
            thre=1e-10,
            thre_inner=1e-10,
            thre_var_est=1e-4,
            pbar=False,
        )
        model.fit(
            df["X"].values, df["S"].values, df["W"].values, df["Y"].values
        )

        params_hist = model._estimator.params_hist_
        R = model._estimator._R

        model.params_.to_csv("./temp_params.csv")
        params_hist.to_csv("./temp_hist.csv")
        np.save("./temp_R.npy", R)

    params = pd.read_csv("./temp_params.csv", index_col=0)
    params_hist = pd.read_csv("./temp_hist.csv", index_col=0)
    R = np.load("./temp_R.npy")
    print(params)

    fig, axs = plt.subplots(ncols=2, figsize=(8, 4), sharey=False)
    params_hist["beta_x"].plot(ax=axs[0])
    df = pd.DataFrame(R[:, 14, :], columns=params_hist.columns)
    df.plot(ax=axs[1])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
