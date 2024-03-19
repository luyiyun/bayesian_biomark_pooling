import logging

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from bayesian_biomarker_pooling.simulate import Simulator
from bayesian_biomarker_pooling.embp import EMBP

# logger = logging.getLogger("EMBP")
# logger.setLevel(logging.DEBUG)
# for handler in logger.handlers:
#     if isinstance(handler, logging.StreamHandler):
#         handler.setLevel(logging.DEBUG)

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
    model.fit(df["X"].values, df["S"].values, df["W"].values, df["Y"].values)
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
# print(
#     simulator.parameters["beta_x"],
#     model.params_.loc["beta_x", "estimate"],
#     # model.params_.loc["beta_x", "variance"],
# )
# print(model.params_)

# params_hist = model._estimator.params_hist_
# R = model._estimator._R

# # params_hist.plot()
# # plt.show()

# fig, axs = plt.subplots(
#     ncols=R.shape[2], nrows=R.shape[1], figsize=(30, 30), sharey=False
# )
# for i in range(R.shape[1]):
#     for j in range(R.shape[2]):
#         ax = axs[i, j]
#         ax.plot(np.arange(R.shape[0]), R[:, i, j])
#         ax.set_title(f"DM(f{i+1, j+1})")
# plt.show()

# simulator = Simulator(type_outcome="binary", n_knowX_per_studies=10)
# df = simulator.simulate()
# print(df)
# model = EMBP(outcome_type="binary")
# model.fit(df["X"].values, df["S"].values, df["W"].values, df["Y"].values)
# print(simulator.parameters["beta_x"], model.params_["beta_x"])
