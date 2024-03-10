from typing import Optional, Dict

import numpy as np
import pandas as pd


class BiomarkerPoolBase:

    def __init__(self) -> None:
        raise NotImplementedError

    def fit(
        X: np.ndarray,
        S: np.ndarray,
        W: np.ndarray,
        Y: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> None:
        raise NotImplementedError

    def summary() -> pd.DataFrame:
        raise NotImplementedError


def check_split_data(
    X: np.ndarray,
    S: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    Z: Optional[np.ndarray] = None,
) -> Dict:
    for arr in [X, S, W, Y]:
        assert arr.ndim == 1
    if Z is not None:
        assert Z.ndim == 2

    ind_studies = np.unique(S)
    ind_xUnKnow = pd.isnull(S)
    ind_xKnow = np.logical_not(ind_xUnKnow)
    n_studies = int(ind_studies.shape[0])
    n_xKnow = int(ind_xKnow.sum())
    n_xUnKnow = int(ind_xUnKnow.sum())

    X_Know = X[ind_xKnow]

    XWYZ_xKnow, WYZ_xUnKnow = {}, {}
    for si in ind_studies:
        ind_si = (ind_studies == si) & ind_xKnow
        if ind_si.sum() > 0:
            XWYZ_xKnow[si] = {"X": X[ind_si], "W": W[ind_si], "Y": Y[ind_si]}
            if Z is not None:
                XWYZ_xKnow[si]["Z"] = Z[ind_si, :]

        ind_si_n = (ind_studies == si) & ind_xUnKnow
        if ind_si_n.sum() > 0:
            WYZ_xUnKnow[si] = {"W": W[ind_si_n], "Y": Y[ind_si_n]}
            if Z is not None:
                WYZ_xUnKnow[si]["Z"] = Z[ind_si_n, :]

    return {
        "n_studies": n_studies,
        "n_xKnow": n_xKnow,
        "n_xUnknow": n_xUnKnow,
        "X_Know": X_Know,
        "XWYZ_xKnow": XWYZ_xKnow,
        "WYZ_xUnKonw": WYZ_xUnKnow
    }
