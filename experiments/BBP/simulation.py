import logging
import multiprocessing as mp
import os
import os.path as osp
import re
import shutil
from argparse import ArgumentParser, Namespace
from typing import List, Tuple
from datetime import datetime
from dataclasses import asdict
from time import perf_counter

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from bayesian_biomarker_pooling import BBP
from bayesian_biomarker_pooling.simulate import Simulator


# suppress the pymc messages
for name in [
    "pymc",
    "main.simulate",
]:
    logger_i = logging.getLogger(name)
    logger_i.setLevel(logging.ERROR)
    logger_i.propagate = False

logger_main = logging.getLogger("main")
logger_main.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s")
ch.setFormatter(formatter)
logger_main.addHandler(ch)


def evaluate(
    true_params: dict[str, float],
    estimate_params: pd.DataFrame,
    interval_columns: Tuple[str, str] = ("hdi_2.5%", "hdi_97.5%"),
) -> pd.DataFrame:
    res = {}
    for k, v in true_params.items():
        assert k in estimate_params.index.levels[1], f"{k} not in estimate_params"
        resi = {}
        bias = estimate_params.loc[(slice(None), k), "mean"].values - v
        resi["bias"] = bias.mean()
        resi["bias_std"] = bias.std()
        resi["percent_bias"] = resi["bias"] / v
        resi["percent_bias_std"] = resi["bias_std"] / v
        resi["mse"] = (bias**2).mean()
        resi["cov_rate"] = (
            (estimate_params.loc[(slice(None), k), interval_columns[0]] <= v)
            & (estimate_params.loc[(slice(None), k), interval_columns[1]] >= v)
        ).mean()
        res[k] = resi

        if "time" in estimate_params.index.levels[1]:
            resi["time"] = estimate_params.loc[
                (slice(None), "time"), "mean"
            ].values.mean()
            resi["time_std"] = estimate_params.loc[
                (slice(None), "time"), "mean"
            ].values.std()

    return pd.DataFrame.from_records(res).T


def run_bbp(
    seed: int, arr: np.ndarray, cols: List[str], z_cols: List[str], args: Namespace
) -> pd.DataFrame:
    dfi = pd.DataFrame(arr, columns=cols)
    dfi["S"] = dfi["S"] - 1  # 模拟实验是从1开始的，现在改成0
    model = BBP(
        seed=seed,
        nsample=args.ndraws,
        ntunes=args.ntunes,
        pbar=False,
        nchains=1,
        prior_betax_args=args.prior_betax_args,
        prior_sigma_dist=args.prior_sigma_dist,
        prior_sigma_args=args.prior_sigma_args,
        prior_a_dist=args.prior_a_dist,
        prior_a_args=args.prior_a_args,
        prior_b_dist=args.prior_b_dist,
        prior_b_args=args.prior_b_args,
        prior_beta0_dist=args.prior_beta0_dist,
        prior_beta0_args=args.prior_beta0_args,
        prior_betaz_args=args.prior_betaz_args,
        prior_x_dist="normal-normal-halfcauchy"
        if args.prior_x == "default"
        else "normal",
        prior_x_args={
            "default": (0, 10.0, 1.0),
            "non-informative": (0, 10.0),
            "informative": None,
        }[args.prior_x],
        solver=args.solver,
    )
    t1 = perf_counter()
    model.fit(dfi, Z_col=z_cols, nuts={"target_accept": 0.99})
    t2 = perf_counter()
    resi = model.summary(
        var_names=["betax", "betaz"] if z_cols is not None else ["betax"]
    )
    resi = pd.concat([resi, pd.DataFrame({"mean": t2 - t1}, index=["time"])], axis=0)
    return resi


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--tasks",
        type=str,
        choices=["simulate", "summarize", "all"],
        default="all",
    )
    parser.add_argument("--save_root", type=str, default="./results/")
    parser.add_argument(
        "--save_action", choices=["cover", "raise", "ignore"], default="raise"
    )
    parser.add_argument("--summarize_target_dir", type=str, default="./results/")
    parser.add_argument("--summarize_save_fn", type=str, default=None)
    parser.add_argument("--summarize_save_sheet", type=str, default="sheet1")

    parser.add_argument("--nrepeat", type=int, default=10)
    parser.add_argument("--ncores", type=int, default=1)

    # simulation settings
    parser.add_argument(
        "--simu_fn", type=str, default=None, help="simulation data file"
    )
    parser.add_argument(
        "--prevalence", type=float, default=0.05, help="prevalence of the disease"
    )
    parser.add_argument(
        "--OR", type=float, default=1.25, help="Odds-ratio of the disease"
    )
    parser.add_argument("--sigma2x", type=float, default=1.0)
    parser.add_argument(
        "--sigma2e", type=float, nargs="+", default=[1.0, 1.0, 1.0, 1.0]
    )
    parser.add_argument("--a", type=float, nargs="+", default=[-3, 1, -1, 3])
    parser.add_argument("--b", type=float, nargs="+", default=[0.5, 0.75, 1.25, 1.5])
    parser.add_argument(
        "--nSamples", type=int, nargs="+", default=[1000, 1000, 1000, 1000]
    )
    parser.add_argument("--nKnowX", type=int, nargs="+", default=[100, 100, 100, 100])
    parser.add_argument(
        "--n_knowX_balance",
        action="store_true",
        help=(
            "if set n_knowX_balance, the y of samples which have X will be balanced."
        ),
    )
    parser.add_argument(
        "--direction", type=str, choices=["w->x", "x->w"], default="x->w"
    )
    parser.add_argument(
        "--betaz",
        type=float,
        nargs="+",
        default=None,
    )

    # bayesian inference settings
    parser.add_argument("--prior_betax_args", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument(
        "--prior_sigma_dist",
        type=str,
        choices=["halfcauchy", "invgamma", "invgamma-gamma-gamma"],
        default="invgamma",
    )
    parser.add_argument("--prior_sigma_args", type=float, nargs="+", default=(2.0, 1.0))
    parser.add_argument(
        "--prior_a_dist",
        type=str,
        default="normal-normal-halfcauchy",
        choices=["normal", "normal-normal-halfcauchy", "normal-normal-invgamma"],
    )
    parser.add_argument(
        "--prior_a_args",
        type=float,
        nargs="+",
        default=[0.0, 10.0, 1.0],
    )
    parser.add_argument(
        "--prior_b_dist",
        type=str,
        default="normal-normal-halfcauchy",
        choices=["normal", "normal-normal-halfcauchy", "normal-normal-invgamma"],
    )
    parser.add_argument(
        "--prior_b_args",
        type=float,
        nargs="+",
        default=[0.0, 10.0, 1.0],
    )
    parser.add_argument(
        "--prior_beta0_dist",
        type=str,
        default="normal-normal-halfcauchy",
        choices=["normal", "normal-normal-halfcauchy", "normal-normal-invgamma"],
    )
    parser.add_argument(
        "--prior_beta0_args",
        type=float,
        nargs="+",
        default=[0.0, 10.0, 1.0],
    )
    parser.add_argument(
        "--prior_betaz_args",
        type=float,
        nargs=2,
        default=[0.0, 10.0],
    )
    parser.add_argument(
        "--prior_x",
        type=str,
        choices=["non-informative", "informative", "default"],
        default="default",
    )
    parser.add_argument("--ndraws", type=int, default=1000)
    parser.add_argument("--ntunes", type=int, default=1000)
    parser.add_argument(
        "--solver", type=str, choices=["pymc", "blackjax"], default="pymc"
    )
    parser.add_argument("--block_size", type=int, default=500)
    parser.add_argument(
        "--pytensor_cache", type=str, default=osp.expanduser("~/.pytensor")
    )
    args = parser.parse_args()

    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)

    # 这些参数的长度需要保持一致或者是scalar
    n_studies = None
    for arg_i in [args.a, args.b, args.sigma2e]:
        if len(arg_i) != 1:
            if n_studies is None:
                n_studies = len(arg_i)
            else:
                assert len(arg_i) == n_studies

    if args.tasks == "summarize":
        all_res = []
        for fn in os.listdir(args.summarize_target_dir):
            if not (fn.endswith(".csv") and fn.startswith("summary_")):
                continue
            prev_i = float(re.search(r"prev([0-9.]*?)_", fn).group(1))
            or_i = float(re.search(r"OR([0-9.]*?)_", fn).group(1))
            summ_df = pd.read_csv(osp.join(args.summarize_target_dir, fn), index_col=0)
            summ_df["prev"] = prev_i
            summ_df["OR"] = or_i
            all_res.append(summ_df)
        all_res = pd.concat(all_res, axis=0).reset_index(names="param")
        all_res.set_index(["param", "prev", "OR"], inplace=True)
        all_res.sort_index(inplace=True)
        print(all_res.to_string())
        if args.summarize_save_fn is not None:
            with pd.ExcelWriter(
                args.summarize_save_fn,
                mode="a" if osp.exists(args.summarize_save_fn) else "w",
            ) as writer:
                all_res.to_excel(writer, sheet_name=args.summarize_save_sheet)
        return

    # 模拟数据
    if args.simu_fn is None:
        name = f"OR{args.OR:.2f}_prev{args.prevalence:.2f}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        simu_fn = osp.join(save_root, f"simu_{name}.h5")
        simulator = Simulator(
            prevalence=args.prevalence,
            OR=args.OR,
            direction=args.direction,
            sigma2_e=args.sigma2e,
            sigma2_x=args.sigma2x,
            a=args.a,
            b=args.b,
            n_sample_per_studies=args.nSamples,
            n_knowX_per_studies=args.nKnowX,
            n_knowX_balance=args.n_knowX_balance,
            betaz=args.betaz,
        )
        arr = []
        # 模拟数据足够快，不需要进行multiprocessing
        for i in tqdm(range(args.nrepeat), desc=f"Simulate {name}: "):
            sim_dat = simulator.simulate(i)
            arr.append(sim_dat.values.astype(float))
        arr = np.stack(arr, axis=0)
        cols = sim_dat.columns.tolist()
        true_betax = simulator.beta1
        true_betaz = simulator.betaz
        with h5py.File(simu_fn, "w") as h5:
            g_sim = h5.create_group("simulate")
            g_sim.attrs["columns"] = cols
            for k, v in asdict(simulator).items():
                g_sim.attrs[k] = "none" if v is None else v
            g_sim.create_dataset("values", data=arr)
        if args.tasks == "simulate":
            return
    else:
        name = re.search(r"simu_(.*?).h5", args.simu_fn).group(1)
        with h5py.File(args.simu_fn, "r") as h5:
            arr = h5["simulate"]["values"][:]
            cols = h5["simulate"].attrs["columns"]
            true_betax = np.log(h5["simulate"].attrs["OR"])
            true_betaz = h5["simulate"].attrs["betaz"]

    # 找出所有的协变量z的列名
    z_cols = [coli for coli in cols if coli.startswith("Z")]
    z_cols = z_cols or None

    # BBP实验
    if args.ncores <= 1:
        esti_res = []
        for i in tqdm(range(args.nrepeat), desc=f"BBP {name}: "):
            resi = run_bbp(i, arr[i], cols, z_cols, args)
            esti_res.append(resi)
    else:
        logger_main.info("Task name: %s" % name)

        def _remove_cache(cache_dir: str, bar=None):
            # 移除pytensor创建的临时文件，避免多进程时的报错
            if cache_dir is not None and osp.exists(cache_dir):
                for fn in os.listdir(cache_dir):
                    msg = "remove pytensor cache: %s" % fn
                    if bar is None:
                        logger_main.info(msg)
                    else:
                        bar.write(msg)
                    shutil.rmtree(osp.join(cache_dir, fn))

        def _mp_block(start_ind: int, end_ind: int, bar=None):
            summ_res = []
            with mp.Pool(args.ncores) as pool:
                temp_reses = [
                    pool.apply_async(
                        run_bbp,
                        (
                            i,
                            arr[i],
                            cols,
                            z_cols,
                            args,
                        ),
                    )
                    for i in range(start_ind, end_ind)
                ]
                for temp_resi in temp_reses:
                    summ_resi = temp_resi.get()
                    summ_res.append(summ_resi)
                    if bar is not None:
                        bar.update()
            return summ_res

        with tqdm(desc="BBP (multi-processing): ", total=args.nrepeat) as bar:
            if args.block_size is None:
                _remove_cache(args.pytensor_cache, bar)
                esti_res = _mp_block(0, args.nrepeat, bar)
            else:
                # 如果nrepeat < block_size，则需要设置为1
                n_block = max((args.nrepeat + 1) // args.block_size, 1)
                esti_res = []
                for bi in range(n_block):
                    _remove_cache(args.pytensor_cache, bar)
                    esti_res_bi = _mp_block(
                        bi * args.block_size,
                        min((bi + 1) * args.block_size, args.nrepeat),
                        bar,
                    )
                    esti_res.extend(esti_res_bi)

    esti_res = pd.concat(esti_res, axis=0, keys=list(range(args.nrepeat)))
    true_values = {"betax": true_betax}
    if z_cols is not None:
        for i, zi in enumerate(true_betaz):
            true_values[f"betaz[{i}]"] = zi
    summ_res = evaluate(true_values, esti_res)
    print(summ_res)
    summ_res.to_csv(osp.join(save_root, f"summary_{name}.csv"))


if __name__ == "__main__":
    main()
