from typing import Any, Dict, List, Optional, Tuple
from collections.abc import Sequence as Seq

import h5py
import numpy as np


def save_h5(
    fn: str,
    arr: np.ndarray,
    group: Optional[str] = None,
    **kwargs: Dict[str, List]
) -> None:
    with h5py.File(fn, "a") as h5:
        if group is not None:
            if group in h5:
                g = h5[group]
            else:
                g = h5.create_group(group)
        else:
            g = h5
        if "values" in g:
            del g["values"]
        g.create_dataset("values", data=arr)
        for k, v in kwargs.items():
            g.attrs[k] = list(v) if isinstance(v, Seq) else v


def load_h5(fn: Tuple[str, h5py.Group]) -> Tuple[np.ndarray, Dict[str, Any]]:
    if isinstance(fn, str):
        with h5py.File(fn, "r") as h5:
            arr = h5["values"][:]
            params = {}
            for k, v in h5.attrs.items():
                params[k] = v
    else:
        arr = fn["values"][:]
        params = {}
        for k, v in fn.attrs.items():
            params[k] = v
    return arr, params
