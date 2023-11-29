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
logger_pymc = logging.getLogger("pymc")
logger_pymc.setLevel(logging.ERROR)

logger_main = logging.getLogger("main")
logger_main.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
ch.setFormatter(formatter)
logger_main.addHandler(ch)


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
        seed=seed,
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
        for k, v in kwargs.items():
            h5.attrs[k] = v


def load_h5(
    fn: str,
) -> Tuple[np.ndarray, Sequence[str], Sequence[str], Dict[str, Any]]:
    with h5py.File(fn, "r") as h5:
        arr = h5["values"][:]
        index = h5.attrs["index"]
        columns = h5.attrs["columns"]
        params = {}
        for k, v in h5.attrs.items():
            if k not in ["index", "columns"]:
                params[k] = v
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--save_action", choices=["cover", "raise", "ignore"], default="raise"
    )

    parser.add_argument("--nrepeat", type=int, default=100)
    parser.add_argument("--ncores", type=int, default=12)

    parser.add_argument(
        "--prevalence", type=float, nargs="+", default=[0.05, 0.25, 0.50]
    )
    parser.add_argument(
        "--OR", type=float, nargs="+", default=[1.25, 1.5, 1.75, 2, 2.25, 2.5]
    )
    args = parser.parse_args()

    save_root = "./results/"
    os.makedirs(save_root, exist_ok=True)

    for prev_i, or_i in product(args.prevalence, args.OR):
        save_name = ("prev%.2f-OR%.2f" % (prev_i, or_i)).replace(".", "_")
        save_name = osp.join(save_root, save_name + ".h5")
        if osp.exists(save_name):
            if args.save_action == "raise":
                raise FileExistsError("%s existed." % save_name)
            elif args.save_action == "ignore":
                logger_main.info(
                    "%s existed, continue next trial." % save_name
                )
                continue
            elif args.save_action == "cover":
                logger_main.info("%s existed, it will be covered." % save_name)
            else:
                raise ValueError

        arr, index, columns = parallel_run(
            range(args.nrepeat),
            ncores=args.ncores,
            prevalence=prev_i,
            OR=or_i,
        )
        print(summarise_results(arr, index, columns))
        save_h5(
            save_name,
            arr,
            index,
            columns,
            prevalence=prev_i,
            OR=or_i,
        )
