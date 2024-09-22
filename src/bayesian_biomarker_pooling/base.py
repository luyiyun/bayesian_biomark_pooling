from typing import Optional, Dict, Tuple, Union, Sequence

import numpy as np
import pandas as pd


class BiomarkerPoolBase:

    def __init__(self) -> None:
        raise NotImplementedError

    def fit(
        df: pd.DataFrame,
        X_col: str = "X",
        S_col: str = "S",
        W_col: str = "W",
        Y_col: str = "Y",
        Z_col: Optional[Union[str, Sequence[str]]] = None,
    ) -> None:
        raise NotImplementedError

    def summary() -> pd.DataFrame:
        raise NotImplementedError


def check_data(
    df: pd.DataFrame,
    X_col: str = "X",
    S_col: str = "S",
    W_col: str = "W",
    Y_col: str = "Y",
    Z_col: Optional[Union[str, Sequence[str]]] = None,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]
]:
    X, S, W, Y = (
        df[X_col].values,
        df[S_col].values,
        df[W_col].values,
        df[list(Y_col) if isinstance(Y_col, Sequence) else Y_col].values,
    )

    if Z_col is not None:
        Z = df[Z_col].values
        assert Z.ndim in (1, 2)
        if Z.ndim == 1:
            Z = Z[:, None]
    else:
        Z = None

    return X, S, W, Y, Z


def split_data(
    X: np.ndarray,
    S: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    Z: Optional[np.ndarray] = None,
) -> Dict:

    studies, ind_s_inv = np.unique(S, return_inverse=True)
    is_m = pd.isnull(X)
    is_o = np.logical_not(is_m)
    n_studies = len(studies)
    n_xKnow = int(is_o.sum())
    n_xUnKnow = int(is_m.sum())

    XWYZ_xKnow, WYZ_xUnKnow, n_ms, n_os = [], [], [], []
    ind_s = []
    for si in studies:
        ind_si = np.nonzero((S == si) & is_o)[0]
        n_os.append(len(ind_si))
        if ind_si.sum() == 0:
            XWYZ_xKnow.append(None)
        else:
            item = [ind_si, X[ind_si], W[ind_si], Y[ind_si], None]
            if Z is not None:
                item[-1] = Z[ind_si, :]
            XWYZ_xKnow.append(item)

        ind_si_n = np.nonzero((S == si) & is_m)[0]
        n_ms.append(len(ind_si_n))
        if ind_si_n.sum() == 0:
            WYZ_xUnKnow.append(None)
        else:
            item = [ind_si_n, W[ind_si_n], Y[ind_si_n], None]
            if Z is not None:
                item[-1] = Z[ind_si_n, :]
            WYZ_xUnKnow.append(item)

        ind_s.append(np.nonzero(S == si)[0])

    return {
        "ind_studies": studies,
        "n_studies": n_studies,
        "n_xKnow": n_xKnow,
        "n_xUnknow": n_xUnKnow,
        "n_ms": np.array(n_ms),
        "n_os": np.array(n_os),
        "ind_o": np.nonzero(is_o)[0],
        "ind_s": ind_s,
        "ind_s_inv": ind_s_inv,
        "XWYZ_xKnow": XWYZ_xKnow,
        "WYZ_xUnKnow": WYZ_xUnKnow,
    }
