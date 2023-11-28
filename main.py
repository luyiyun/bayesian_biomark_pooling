import logging
import multiprocessing as mp
import os
import os.path as osp
from argparse import ArgumentParser
from itertools import product
from typing import Any, Dict, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluate import evaluate
from src.method import bayesian_analysis
from src.simulate import simulate

# suppress the pymc messages
logger = logging.getLogger("pymc")
logger.setLevel(logging.ERROR)


def simulate_bayesian_evaluation(
    seed: int, prevalence: float, OR: float
) -> pd.DataFrame:
    sim_dat, true_params = simulate(
        seed=seed, prevalence=prevalence, beta1=np.log(OR)
    )
    baye_res = bayesian_analysis(
        sim_dat,
        pbar=False,
        solver="mcmc",
        var_names=["a", "b", "a_s", "b_s", "beta0", "betax"],
        nchains=1,
    )
    eval_res = evaluate(true_params, baye_res)
    return eval_res


def parallel_run(
    seeds: Sequence[int],
    ncores: int = 4,
    prevalence: float = 0.5,
    OR: float = 1.25,
) -> Tuple[np.ndarray, pd.Index, pd.Index]:
    all_res = []
    if ncores <= 1:
        for seedi in seeds:
            resi = simulate_bayesian_evaluation(seedi)
            all_res.append(resi.values)
    else:
        with mp.Pool(ncores) as pool:
            temp_reses = [
                pool.apply_async(
                    simulate_bayesian_evaluation, (seedi, prevalence, OR)
                )
                for seedi in seeds
            ]
            for temp_resi in tqdm(
                temp_reses, desc="prev=%.2f,OR=%.2f" % (prevalence, OR)
            ):
                resi = temp_resi.get()
                all_res.append(resi.values)
    all_res = np.stack(all_res, axis=0)
    return all_res, resi.index, resi.columns


def save_h5(
    fn: str,
    arr: np.ndarray,
    index: Sequence[str],
    columns: Sequence[str],
    **kwargs: Dict[str, Any]
) -> None:
    with h5py.File(fn, "w") as h5:
        h5.create_dataset("values", data=arr)
        h5.attrs["index"] = list(index)
        h5.attrs["columns"] = list(columns)
        h5.attrs["params"] = kwargs


def load_h5(
    fn: str,
) -> Tuple[np.ndarray, Sequence[str], Sequence[str], Dict[str, Any]]:
    with h5py.File(fn, "r") as h5:
        arr = h5["values"][:]
        index = h5.attrs["index"]
        columns = h5.attrs["columns"]
        params = h5.attrs["params"]
    return arr, index, columns, params


def summarise_results(
    arr: np.ndarray, index: Sequence[str], columns: Sequence[str]
) -> pd.DataFrame:
    arr_mean, arr_std = arr.mean(axis=0), arr.std(axis=0)
    res = np.full_like(arr_mean, "", dtype="U25")
    for i in range(arr_mean.shape[0]):
        for j in range(arr_mean.shape[1]):
            res[i, j] = "%.4f±%.4f" % (arr_mean[i, j], arr_std[i, j])
    return pd.DataFrame(res, index=index, columns=columns)


def main():
    parser = ArgumentParser()
    parser.add_argument("--nrepeat", type=int, default=100)
    parser.add_argument("--ncores", type=int, default=12)

    parser.add_argument("--prevalence", type=float, nargs="+", default=[0.05])
    parser.add_argument("--OR", type=float, nargs="+", default=[1.25])
    args = parser.parse_args()

    save_root = "./results/"
    os.makedirs(save_root, exist_ok=True)

    for prev_i, or_i in product(args.prevalence, args.OR):
        arr, index, columns = parallel_run(
            range(args.nrepeat),
            ncores=args.ncores,
            prevalence=prev_i,
            OR=or_i,
        )
        print(summarise_results(arr, index, columns))
        save_name = ("prev%.2f-OR%.2f" % (prev_i, or_i)).replace(".", "_")
        save_h5(
            osp.join(save_root, save_name + ".h5"),
            arr,
            index,
            columns,
            prevalence=prev_i,
            OR=or_i,
        )

    # arr, index, columns = load_h5(osp.join(save_root, "test.h5"))
    # print(summarise_results(arr, index, columns))


if __name__ == "__main__":
    main()
