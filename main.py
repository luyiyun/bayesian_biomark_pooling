import logging
import multiprocessing as mp
import os
import os.path as osp
import shutil
from argparse import ArgumentParser
from itertools import product
from typing import Dict, List, Literal, Optional, Sequence, Tuple

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

logger_simu = logging.getLogger("main.simulate")
logger_simu.setLevel(logging.ERROR)

logger_main = logging.getLogger("main")
logger_main.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
ch.setFormatter(formatter)
logger_main.addHandler(ch)


def _pipeline(
    seed: int,
    simulate_kwargs: Dict = {},
    analysis_kwargs: Dict = {},
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sim_dat, true_params = simulate(seed=seed, **simulate_kwargs)
    sim_dat["S"] = sim_dat["S"] - 1  # 模拟实验是从1开始的，现在改成0
    baye_res = bayesian_analysis(sim_dat, seed=seed, **analysis_kwargs)
    eval_res = evaluate(true_params, baye_res)
    # 使用ndarray和list会提高多线程的效率
    return (
        (sim_dat.values.astype(float), sim_dat.columns.tolist()),
        (
            baye_res.values.astype(float),
            baye_res.columns.tolist(),
            baye_res.index.tolist(),
        ),
        (
            eval_res.values.astype(float),
            eval_res.columns.tolist(),
            eval_res.index.tolist(),
        ),
    )


class Trials:
    def __init__(
        self,
        nrepeat: int,
        ncores: int = 1,
        prevalence: float = 0.05,
        OR: float = 1.25,
        direction: Literal["x->w", "w->x"] = "x->w",
        solver: Literal["pymc", "blackjax", "numpyro", "vi"] = "pymc",
        pytensor_cache: Optional[str] = None,
    ) -> None:
        self._nrepeat = nrepeat
        self._ncores = ncores
        self._simulate_kwargs = dict(
            prevalence=prevalence, beta1=np.log(OR), direction=direction
        )
        self._analysis_kwargs = dict(solver=solver)
        self._name = (
            "prev%.2f-OR%.2f-direct[%s]" % (prevalence, OR, direction)
        ).replace(".", "_")
        self._pytensor_cache = pytensor_cache

    def simulate(
        self, nrepeat: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        arr = []
        for i in tqdm(
            range(nrepeat if nrepeat is not None else self._nrepeat),
            desc="Simulate %s: " % self._name,
        ):
            sim_dat, _ = simulate(seed=i, **self._simulate_kwargs)
            arr.append(sim_dat.values.astype(float))
        arr = np.stack(arr, axis=0)
        columns = sim_dat.columns.tolist()
        return arr, columns

    def pipeline(
        self, nrepeat: Optional[int] = None, ncores: Optional[int] = None
    ):
        nrepeat = nrepeat if nrepeat is not None else self._nrepeat
        ncores = ncores if ncores is not None else self._ncores
        res_simu, res_anal, res_eval = [], [], []
        if ncores <= 1:
            for i in tqdm(
                range(nrepeat),
                desc="Pipeline %s: " % self._name,
            ):
                (
                    (sim_i, sim_col),
                    (ana_i, ana_col, ana_ind),
                    (eva_i, eva_col, eva_ind),
                ) = _pipeline(i, self._simulate_kwargs, self._analysis_kwargs)
                res_simu.append(sim_i)
                res_anal.append(ana_i)
                res_eval.append(eva_i)
        else:
            # 移除pytensor创建的临时文件，避免多进程时的报错
            if self._pytensor_cache is not None and osp.exists(
                self._pytensor_cache
            ):
                for fn in os.listdir(self._pytensor_cache):
                    logger_main.info("remove pytensor cache: %s" % fn)
                    shutil.rmtree(osp.join(self._pytensor_cache, fn))

            with mp.Pool(ncores) as pool:
                temp_reses = [
                    pool.apply_async(
                        _pipeline, (seedi, self._simulate_kwargs, {})
                    )
                    for seedi in range(nrepeat)
                ]
                for temp_resi in tqdm(
                    temp_reses, desc="Pipeline %s: " % self._name
                ):
                    (
                        (sim_i, sim_col),
                        (ana_i, ana_col, ana_ind),
                        (eva_i, eva_col, eva_ind),
                    ) = temp_resi.get()
                    res_simu.append(sim_i)
                    res_anal.append(ana_i)
                    res_eval.append(eva_i)

        res_simu = np.stack(res_simu, axis=0)
        res_anal = np.stack(res_anal, axis=0)
        res_eval = np.stack(res_eval, axis=0)
        return (
            (res_simu, sim_col),
            (res_anal, ana_col, ana_ind),
            (res_eval, eva_col, eva_ind),
        )


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
        "--tasks",
        type=str,
        choices=["simulate", "summarize", "all"],
        default="all",
    )
    parser.add_argument(
        "--save_action", choices=["cover", "raise", "ignore"], default="raise"
    )

    parser.add_argument("--nrepeat", type=int, default=100)
    parser.add_argument("--ncores", type=int, default=12)
    parser.add_argument(
        "--solver",
        type=str,
        choices=["pymc", "blackjax", "numpyro", "vi"],
        default="pymc",
    )

    parser.add_argument(
        "--prevalence", type=float, nargs="+", default=[0.05, 0.25, 0.50]
    )
    parser.add_argument(
        "--OR", type=float, nargs="+", default=[1.25, 1.5, 1.75, 2, 2.25, 2.5]
    )
    parser.add_argument(
        "--direction", type=str, choices=["w->x", "x->w"], default="x->w"
    )
    args = parser.parse_args()

    save_root = "./results/"
    os.makedirs(save_root, exist_ok=True)

    for prev_i, or_i in product(args.prevalence, args.OR):
        trial_i = Trials(
            nrepeat=args.nrepeat,
            ncores=args.ncores,
            prevalence=prev_i,
            OR=or_i,
            direction=args.direction,
            solver=args.solver,
            pytensor_cache="/home/rongzhiwei/.pytensor/",
        )
        if args.tasks == "simulate":
            save_fn = osp.join(save_root, "simulate_%s.h5" % trial_i._name)
            arr, cols = trial_i.simulate()
            with h5py.File(save_fn, "w") as h5:
                g_sim = h5.create_group("simulate")
                g_sim.attrs["columns"] = cols
                g_sim.create_dataset("values", data=arr)
        elif args.tasks == "summarize":
            res_fn = osp.join(save_root, "pipeline_%s.h5" % trial_i._name)
            with h5py.File(res_fn, "a") as h5:
                g_eva = h5["evaluate"]
                print(
                    summarise_results(
                        g_eva["values"][:],
                        g_eva.attrs["index"],
                        g_eva.attrs["columns"],
                    ),
                )
        elif args.tasks == "all":
            save_fn = osp.join(save_root, "pipeline_%s.h5" % trial_i._name)
            if osp.exists(save_fn):
                if args.save_action == "raise":
                    raise FileExistsError(save_fn)
                elif args.save_action == "ignore":
                    logger_main.info("%s existed, skip." % save_fn)
                    continue
            (
                (sim_i, sim_col),
                (ana_i, ana_col, ana_ind),
                (eva_i, eva_col, eva_ind),
            ) = trial_i.pipeline()
            with h5py.File(save_fn, "w") as h5:
                g_sim = h5.create_group("simulate")
                g_sim.attrs["columns"] = sim_col
                g_sim.create_dataset("values", data=sim_i)
                g_ana = h5.create_group("analysis")
                g_ana.attrs["columns"] = ana_col
                g_ana.attrs["index"] = ana_ind
                g_ana.create_dataset("values", data=ana_i)
                g_eva = h5.create_group("evaluate")
                g_eva.attrs["columns"] = eva_col
                g_eva.attrs["index"] = eva_ind
                g_eva.create_dataset("values", data=eva_i)
            print(summarise_results(eva_i, eva_ind, eva_col))
