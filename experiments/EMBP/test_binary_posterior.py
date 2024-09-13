import numpy as np
# import pandas as pd
from numpy import ndarray
from scipy.special import expit  # , ndtri
import scipy.stats as sst
import pymc as pm
# import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def laplacian_approximation(
    x: ndarray,
    y: ndarray,
    niter: int = 100,
    delta1: float = 1e-3,
    delta2: float = 1e-5,
):
    assert x.ndim == 2
    x_des = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    beta = np.ones(x_des.shape[1])
    for i in tqdm(range(niter)):
        p = expit(x_des @ beta)
        grad = x_des.T @ (p - y)
        hess = np.einsum("ij,i,ik->jk", x_des, p * (1 - p), x_des)
        delta_beta = np.linalg.solve(hess, grad)
        rdiff = np.max(np.abs(delta_beta) / (np.abs(beta) + delta1))
        beta -= delta_beta
        if rdiff < delta2:
            break

    p = expit(x_des @ beta)
    # grad = x_des.T @ (p - y)
    hess = np.einsum("ij,i,ik->jk", x_des, p * (1 - p), x_des)

    return beta, hess


n = 100
# beta_0 = [0.0]
# beta_1 = [-1, 0, 1]
# beta_01_pairs = list(product(beta_0, beta_1))
beta_01_pairs = [(-1, 1), (-1, 0), (1, -1), (0, 0)]

npairs = len(beta_01_pairs)
nr = int(np.sqrt(npairs))
nc = (npairs + 1) // nr
fig, axs = plt.subplots(ncols=nc, nrows=nr, figsize=(nc * 3, nr * 3))
axs = axs.flatten()

for i, (beta_0, beta_1) in enumerate(beta_01_pairs):
    print(f"beta_0: {beta_0:.0f}, beta_1: {beta_1:.0f}")

    ax = axs[i]

    # simulate data
    x = np.random.randn(n)
    logit = x * beta_1 + beta_0
    y = np.random.binomial(n=1, p=expit(logit))
    print(f"The proportion of y=1 is {y.mean():.2f}")

    # laplacian approximation
    beta, hess = laplacian_approximation(x[:, None], y)
    hess_inv = np.linalg.inv(hess)
    # sst.multivariate_normal(beta, np.linalg.inv(hess))
    mu, sigma = beta[1], np.sqrt(hess_inv[1, 1])
    norm_beta1 = sst.norm(mu, sigma)
    x_norm_beta1_lap = np.linspace(mu - 4 * sigma, mu + 4 * sigma, num=1000)
    y_norm_beta1_lap = norm_beta1.pdf(x_norm_beta1_lap)

    # MCMC
    with pm.Model() as model:
        a = pm.Flat("beta0")
        b = pm.Flat("beta1")
        pm.Bernoulli("y", logit_p=a + b * x, observed=y)
        res = pm.sample()

    beta1_mcmc = res.posterior["beta1"].data.flatten()

    ax.plot(x_norm_beta1_lap, y_norm_beta1_lap, "-")
    sns.kdeplot(beta1_mcmc, color="red", ax=ax)
    ax.set_title(f"beta_0={beta_0:.0f}, beta_1={beta_1:.0f}")

fig.tight_layout()
fig.savefig("./binary_posterior.png")
