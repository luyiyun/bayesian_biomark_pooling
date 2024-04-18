from typing import Tuple

import numpy as np
from numpy import ndarray
from scipy.special import expit
from scipy.linalg import lstsq

from ..logger import logger_embp


def batch_nonzero(mask):
    if mask.ndim == 1:
        return np.nonzero(mask)[0]
    else:
        return np.arange(mask.shape[0])[:, None], np.stack(
            [np.nonzero(mask[i])[0] for i in range(mask.shape[0])],
        )


def ols(
    X: ndarray, Y: ndarray, Z: ndarray | None = None
) -> Tuple[ndarray, ndarray]:
    X_des = np.stack([X, np.ones_like(X)], axis=-1)
    if Z is not None:
        X_des = np.concatenate([X_des, Z], axis=-1)

    beta, resid, _, _ = lstsq(X_des, Y)
    # x2 = np.einsum("ij,ik->jk", X_des, X_des)
    # x2_inv = np.linalg.inv(x2)
    # hat_mat = np.einsum("ij,kj->ik", x2_inv, X_des)
    # beta = np.einsum("ik,k->i", hat_mat, Y)
    # pred = np.einsum("ij,j->i", X_des, beta)
    # sigma2 = np.mean((Y - pred) ** 2, axis=-1)
    return beta, resid


def logistic(
    X: ndarray,
    Y: ndarray,
    Z: ndarray | None = None,
    lr: float = 1.0,
    delta1: float = 1e-3,
    delta2: float = 1e-7,
    max_iter: int = 100,
) -> ndarray:
    X_des = np.stack([X, np.ones_like(X)], axis=-1)
    if Z is not None:
        X_des = np.concatenate([X_des, Z], axis=-1)

    beta_ = np.zeros(X_des.shape[1])
    for i in range(max_iter):
        p = expit(X_des @ beta_)
        grad = X_des.T @ (p - Y)
        hess = np.einsum("ij,i,ik->jk", X_des, p * (1 - p), X_des)

        beta_delta = lr * np.linalg.solve(hess, grad)
        beta_ -= beta_delta

        rdiff = np.max(np.abs(beta_delta) / (np.abs(beta_) + delta1))
        logger_embp.info(
            f"Init step Newton-Raphson: iter={i+1} diff={rdiff:.4f}"
        )
        if rdiff < delta2:
            break
    else:
        logger_embp.warning(
            f"Init Newton-Raphson (max_iter={max_iter}) doesn't converge"
        )

    return beta_
