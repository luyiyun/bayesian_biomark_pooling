from typing import Dict, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def evaluate(
    true_params: Dict[str, Union[float, Sequence[float]]],
    estimate_params: pd.DataFrame,
    interval_columns: Tuple[str, str] = ("hdi_2.5%", "hdi_97.5%"),
    name_pairs: Sequence[Tuple[str, str]] = (
        ("beta1", "betax"),
        ("a", "a_s"),
        ("b", "b_s"),
    ),
) -> pd.DataFrame:
    index, true_arr = [], []
    for k_tp, k_ep in name_pairs:
        v = true_params[k_tp]
        if isinstance(v, np.ndarray) and v.ndim == 0:
            index.append(k_ep)
            true_arr.append(v.item())
        elif isinstance(v, (float, int)):
            index.append(k_ep)
            true_arr.append(v)
        elif isinstance(v, (list, tuple, np.ndarray)):
            index.extend(["%s[%d]" % (k_ep, i) for i in range(len(v))])
            true_arr.extend(list(v))
        else:
            raise TypeError
    true_arr = np.array(true_arr)
    estimate_params = estimate_params.loc[index, :]

    bias = estimate_params["mean"].values - true_arr
    percent_bias = bias / true_arr
    mse = bias**2
    cov_rate = np.logical_and(
        true_arr >= estimate_params.loc[:, interval_columns[0]].values,
        true_arr <= estimate_params.loc[:, interval_columns[1]].values,
    ).astype(float)

    return pd.DataFrame(
        {
            "bias": bias,
            "percent_bias": percent_bias,
            "mse": mse,
            "cov_rate": cov_rate,
        },
        index=index,
    )
