import logging
import multiprocessing as mp
import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob
from typing import Dict, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import h5py

from bayesian_biomarker_pooling import BBP


logger = logging.getLogger("experiments.BBP.run_simulation")


def bbp_pipeline(
    h5fn: str,
    key: str,
    Z_col: Optional[Sequence[str]],
    bbp_kwargs: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dfi = pd.read_hdf(h5fn, key)
    model = BBP(
        pbar=False,
        nchains=1,
        target_accept=0.95,
        **bbp_kwargs,
    )
    fit_res = model.fit(
        dfi,
        Z_col=Z_col,
        Y_col=("T", "E") if bbp_kwargs["type_outcome"] == "survival" else "Y",
    )
    return fit_res.summary(
        var_names=["betax", "a_s", "b_s", "beta0s"]
        + (["betaz"] if Z_col is not None else [])
    )


def main():

    ##################################################################
    # Command-line arguments
    ##################################################################
    parser = ArgumentParser()
    parser.add_argument("--target_data", type=str)
    parser.add_argument("--save_root", type=str, default="./results/")
    parser.add_argument("--save_prefix", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ncores", type=int, default=1)
    parser.add_argument(
        "--solver", type=str, choices=["pymc", "blackjax"], default="pymc"
    )
    # parser.add_argument("--block_size", type=int, default=500)

    parser.add_argument(
        "--type_outcome",
        type=str,
        choices=["binary", "continue", "survival"],
        default="binary",
    )
    parser.add_argument(
        "--prior_sigma_ws",
        type=str,
        choices=["gamma", "inv_gamma"],
        default="inv_gamma",
    )
    parser.add_argument(
        "--prior_sigma_ab0",
        type=str,
        choices=["half_cauchy", "half_flat"],
        default="half_cauchy",
    )
    parser.add_argument(
        "--prior_betax_std",
        type=lambda x: x if x == "inf" else float(x),
        default="inf",
    )
    parser.add_argument(
        "--prior_betaz_std",
        type=lambda x: x if x == "inf" else float(x),
        default="inf",
    )
    parser.add_argument("--prior_a_std", type=float, default=1.0)
    parser.add_argument("--prior_b_std", type=float, default=1.0)
    parser.add_argument("--prior_beta0_std", type=float, default=1.0)
    parser.add_argument("--ndraws", type=int, default=1000)
    parser.add_argument("--ntunes", type=int, default=1000)

    args = parser.parse_args()

    ##################################################################
    # set logging
    ##################################################################
    logging.basicConfig(
        format="[%(asctime)s][%(name)s][%(levelname)s]: %(message)s",
        level=logging.INFO,
    )
    # silence logger of pymc
    logger_pymc = logging.getLogger("pymc")
    logger_pymc.setLevel(logging.ERROR)
    if args.solver == "blackjax":
        logger_jax = logging.getLogger("jax")
        logger_jax.setLevel(logging.ERROR)

    ##################################################################
    # set the path containing results
    ##################################################################
    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)

    ##################################################################
    # run and evaluate BBP for each seed
    ##################################################################
    dat_files = glob(args.target_data)
    for i, fn in enumerate(dat_files):
        name = osp.basename(fn)[:-3]
        logger.info(f"({i+1}/{len(dat_files)}) Run BBP on {name}")

        # =================== load simulated data ===================
        with h5py.File(fn, "r") as h5f:
            seeds = list(h5f.keys())

        # load simulation parameters
        json_fn = fn[:-2] + "json"
        with open(json_fn, "r") as jf:
            simu_params = json.load(jf)
        true_value = simu_params["beta_x"]
        Z_col = (
            None
            if simu_params["beta_z"] is None
            else [f"Z{i+1}" for i in range(len(simu_params["beta_z"]))]
        )

        # =================== run BBP ===================
        bbp_kwargs = dict(
            seed=args.seed,
            prior_betax_std=args.prior_betax_std,
            prior_betaz_std=args.prior_betaz_std,
            prior_sigma_ws=args.prior_sigma_ws,
            prior_sigma_ab0=args.prior_sigma_ab0,
            std_prior_a=args.prior_a_std,
            std_prior_b=args.prior_b_std,
            std_prior_beta0=args.prior_beta0_std,
            nsample=args.ndraws,
            ntunes=args.ntunes,
            solver=args.solver,
            type_outcome=args.type_outcome,
        )
        if args.ncores <= 1:
            res_bbp = []
            for i in tqdm(seeds):
                resi = bbp_pipeline(fn, i, Z_col, bbp_kwargs)
                # use xarray to construct named multi-dimensional array
                res_bbp.append(xr.DataArray(resi, dims=["param", "stats"]))
        else:
            # def _remove_cache(bar=None):
            #     # 移除pytensor创建的临时文件，避免多进程时的报错
            #     cache_dir = "~/.pytensor/"
            #     if cache_dir is not None and osp.exists(cache_dir):
            #         for fn in os.listdir(cache_dir):
            #             msg = "remove pytensor cache: %s" % fn
            #             if bar is None:
            #                 logger.info(msg)
            #             else:
            #                 bar.write(msg)
            #             shutil.rmtree(osp.join(cache_dir, fn))

            with mp.Pool(args.ncores) as pool:
                temp_reses = [
                    pool.apply_async(
                        bbp_pipeline,
                        (
                            fn,
                            i,
                            Z_col,
                            bbp_kwargs,
                        ),
                    )
                    for i in seeds
                ]
                res_bbp = [
                    xr.DataArray(resi.get(), dims=["param", "stats"])
                    for resi in tqdm(temp_reses)
                ]

        res_bbp = xr.concat(res_bbp, pd.Index(seeds, name="seed"))

        # =================== evaluation ===================

        # calculate the scores
        all_betax = res_bbp.loc[:, "betax", "mean"].values
        hdi_1 = res_bbp.loc[:, "betax", "hdi_2.5%"].values
        hdi_2 = res_bbp.loc[:, "betax", "hdi_97.5%"].values

        bias = all_betax - true_value
        percent_bias = bias / true_value
        mse = bias**2
        cov_rate = np.logical_and(
            true_value >= hdi_1, true_value <= hdi_2
        ).astype(float)

        res_eval = xr.DataArray(
            pd.DataFrame(
                {
                    "bias": bias,
                    "percent_bias": percent_bias,
                    "mse": mse,
                    "cov_rate": cov_rate,
                },
                index=res_bbp.coords["seed"].values,
            ),
            dims=["seed", "scores"],
        )

        res_scores = res_eval.mean("seed")

        # =================== save results of BBP ===================
        res = xr.Dataset(
            {"run": res_bbp, "eval": res_eval, "score": res_scores}
        )
        res.to_netcdf(
            osp.join(
                args.save_root,
                (
                    f"{name}.nc"
                    if args.save_prefix is None
                    else f"{args.save_prefix}_{name}.nc"
                ),
            )
        )
        print(res["score"].to_dataframe())


if __name__ == "__main__":
    main()
