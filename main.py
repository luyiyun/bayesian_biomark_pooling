import logging
import os
import multiprocessing as mp
import re
import json
from datetime import datetime
from typing import Literal, Sequence
from argparse import ArgumentParser
from itertools import product
from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
import statsmodels.api as sm

from bayesian_biomarker_pooling.simulate import Simulator
from bayesian_biomarker_pooling import EMBP


def plot_params_hist(params_hist: np.ndarray, names: np.ndarray, savefn: str):
    nparams = params_hist.shape[1]
    nr = int(np.sqrt(nparams))
    nc = (nparams + nr - 1) // nr
    fig, axs = plt.subplots(
        nrows=nr, ncols=nc, figsize=(nc * 3, nr * 3), sharey=False
    )
    axs = axs.flatten()
    for i in range(nparams):
        pd.Series(params_hist[:, i]).plot(ax=axs[i])
        axs[i].set_title(names[i])
    fig.tight_layout()
    fig.savefig(savefn)


def temp_test_continue(ci=False, ve_method="bootstrap", seed=0):
    root = "./results/tmp_embp"
    os.makedirs(root, exist_ok=True)

    simulator = Simulator(type_outcome="continue")
    df = simulator.simulate(seed)
    model = EMBP(
        outcome_type="continue",
        ci=ci,
        ci_method=ve_method,
        pbar=True,
        seed=seed,
    )
    model.fit(df["X"].values, df["S"].values, df["W"].values, df["Y"].values)

    params = model.params_
    params_hist = model.params_hist_
    plot_params_hist(
        params_hist.values,
        params_hist.columns.values,
        os.path.join(root, "params_hist.png"),
    )
    if ci and (ve_method == "sem"):
        R = model._estimator._R
        np.save("./temp_R.npy", R)
        plot_params_hist(
            R[:, 14, :],
            params_hist.columns.values,
            os.path.join(root, "R_beta_x.png"),
        )

    print(params)


def temp_test_binary(
    ci=False,
    seed=0,
    beta_z=None,
    nsample=100,
    n_knowX=10,
    beta_x=1.0,
    binary_solve="lap",
):
    root = "./results/tmp_embp"
    os.makedirs(root, exist_ok=True)

    simulator = Simulator(
        type_outcome="binary",
        n_sample_per_studies=nsample,
        n_knowX_per_studies=n_knowX,
        sigma2_y=[0.5, 0.75, 1.0, 1.25],
        sigma2_e=[0.5, 0.75, 1.0, 1.25],
        beta_z=beta_z,
        beta_x=beta_x,
    )
    df = simulator.simulate(seed)
    model = EMBP(
        outcome_type="binary",
        ci=ci,
        # variance_estimate_method="bootstrap",
        # max_iter=300,
        pbar=True,
        # n_importance_sampling=100,
        use_gpu=False,
        seed=seed,
        binary_solve=binary_solve,
    )
    model.fit(
        df["X"].values,
        df["S"].values,
        df["W"].values,
        df["Y"].values,
        df.filter(like="Z", axis=1).values if beta_z is not None else None,
    )

    params = model.params_
    params_hist = model.params_hist_
    model.params_.to_csv(os.path.join(root, "binary_params.csv"))
    params_hist.to_csv(os.path.join(root, "binary_hist.csv"))

    plot_params_hist(
        params_hist.values,
        params_hist.columns.values,
        os.path.join(root, "binary_params_hist.png"),
    )

    print(params)


def method_xonly(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray | None,
    type_outcome: Literal["binary", "continue"],
) -> np.ndarray:
    notnone = ~pd.isnull(X)
    X, Y = X[notnone], Y[notnone]
    if Z is not None:
        X = np.concatenate([X[:, None], Z[notnone]], axis=1)
    X = sm.add_constant(X)
    if type_outcome == "continue":
        model = sm.OLS(Y, X)
    else:
        model = sm.GLM(Y, X, sm.families.Binomial())
    res = model.fit()
    return np.r_[res.params[1], res.conf_int()[1, :]]


def method_naive(
    W, Y, Z, type_outcome: Literal["binary", "continue"]
) -> np.ndarray:
    if Z is not None:
        W = np.concatenate([W[:, None], Z], axis=1)
    W = sm.add_constant(W)
    if type_outcome == "continue":
        model = sm.OLS(Y, W)
    else:
        model = sm.GLM(Y, W, sm.families.Binomial())
    res = model.fit()
    return np.r_[res.params[1], res.conf_int()[1, :]]


def trial_once_by_simulator_and_estimator(
    type_outcome: Literal["continue", "binary"],
    simulator: Simulator,
    estimator: EMBP | dict | None = None,
    seed: int | None = None,
    methods: Sequence[Literal["EMBP", "xonly", "naive"]] = (
        "xonly",
        "naive",
        "EMBP",
    ),
) -> dict[str, np.ndarray]:
    # 1. generate data
    df = simulator.simulate(seed=seed)
    zind = df.columns.map(lambda x: re.search(r"Z\d*", x) is not None)
    X = df["X"].values
    Y = df["Y"].values
    W = df["W"].values
    Z = df.loc[:, zind].values if zind.any() else None

    # 2. other methods
    res_all = {}
    for methodi in methods:
        if methodi == "xonly":
            res = method_xonly(X, Y, Z, type_outcome)
        elif methodi == "naive":
            res = method_naive(W, Y, Z, type_outcome)
        elif methodi == "EMBP":
            # 3. run model
            if estimator is None:
                continue
            elif isinstance(estimator, dict):
                estimator = EMBP(**estimator)
            estimator.fit(X, df["S"].values, W, Y, Z)
            res = estimator.params_.values
        res_all[methodi] = res
    return res_all


def main():

    parser = ArgumentParser()

    parser.add_argument(
        "-ot",
        "--outcome_type",
        default="continue",
        choices=["continue", "binary"],
    )
    parser.add_argument(
        "-nsps", "--nsample_per_studies", default=(100,), type=int, nargs="+"
    )
    parser.add_argument(
        "-rxps", "--ratio_x_per_studies", default=(0.1,), type=float, nargs="+"
    )
    parser.add_argument(
        "-bx", "--beta_x", default=(0.0,), type=float, nargs="+"
    )
    parser.add_argument(
        "-b0",
        "--beta_0",
        default=(-0.5, -0.25, 0.25, 0.5),
        type=float,
        nargs=4,
        help="如果设置了prevalence并且outcome_type=binary，则其失效",
    )
    parser.add_argument("-bz", "--beta_z", default=None, type=float, nargs="*")
    parser.add_argument("-pr", "--prevalence", default=None, type=float)

    parser.add_argument(
        "-m",
        "--methods",
        default=("EMBP", "xonly", "naive"),
        nargs="+",
        choices=("EMBP", "xonly", "naive"),
    )
    parser.add_argument("-nr", "--nrepeat", default=1000, type=int)
    parser.add_argument("-nc", "--ncores", default=1, type=int)
    parser.add_argument("-t", "--test", action="store_true")
    parser.add_argument(
        "-l",
        "--log",
        default="warn",
        choices=["error", "warn", "info", "debug"],
    )
    parser.add_argument("--pbar", action="store_true")
    parser.add_argument("--root", default="./results/embp/")
    parser.add_argument("--name", default=None)
    parser.add_argument("--skip", default=None, nargs="*", type=str)

    parser.add_argument("--no_ci", action="store_true")
    parser.add_argument(
        "--ci_method", default="bootstrap", choices=("bootstrap", "sem")
    )
    parser.add_argument("--max_iter", default=None, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n_bootstrap", default=200, type=int)
    parser.add_argument("--gem", action="store_true")
    parser.add_argument("-qK", "--quasi_K", default=100, type=int)
    parser.add_argument(
        "-bs", "--binary_solve", default="lap", choices=["lap", "is"]
    )

    args = parser.parse_args()

    log_level = {
        "error": logging.ERROR,
        "warn": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }[args.log]
    logger = logging.getLogger("EMBP")
    logger.setLevel(log_level)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(log_level)

    if args.test:
        if args.outcome_type == "continue":
            temp_test_continue(ci=not args.no_ci, ve_method="bootstrap")
        elif args.outcome_type == "binary":
            temp_test_binary(
                ci=not args.no_ci,
                seed=1,
                nsample=100,
                n_knowX=10,
                beta_x=args.beta_x[0],
                binary_solve=args.binary_solve,
            )
        return

    # 模拟实验：
    # 1. 不同样本量，不同缺失比例下的效果,
    # 2. 一类错误 & 效能
    # 3. 把参数默认值搞清楚

    # if args.skip_dup:
    #     runned_configs = []
    #     for fn in os.listdir(args.root):
    #         if fn.startswith(
    #             args.outcome_type if args.name is None else args.name
    #         ) and fn.endswith(".json"):
    #             with open(osp.join(args.root, fn), "r") as f:
    #                 runned_configs.append(json.load(f))
    if args.skip is not None:
        skip_set = [
            tuple([float(s) for s in skip_str.split(",")])
            for skip_str in args.skip
        ]
    else:
        skip_set = []

    for i, (ns, rx, bx) in enumerate(
        product(
            args.nsample_per_studies, args.ratio_x_per_studies, args.beta_x
        )
    ):
        print(
            f"nsample per studies: {ns}, "
            f"ratio of observed x: {rx:.2f}, true beta x: {bx:.2f}"
        )
        if (ns, rx, bx) in skip_set:
            print("skip")
            continue

        seedi = args.seed + i
        nx = int(rx * ns)

        json_content = deepcopy(args.__dict__)
        json_content["nsample"] = ns
        json_content["ratiox"] = rx
        json_content["betax"] = bx
        json_content["seed"] = seedi
        json_content["nx"] = nx

        simulator = Simulator(
            type_outcome=args.outcome_type,
            beta_x=bx,
            sigma2_y=[0.5, 0.75, 1.0, 1.25],
            sigma2_e=[0.5, 0.75, 1.0, 1.25],
            beta_0=(
                None
                if args.outcome_type == "binary"
                and args.prevalence is not None
                else args.beta_0
            ),
            prevalence=args.prevalence,
            n_sample_per_studies=ns,
            n_knowX_per_studies=nx,
            beta_z=args.beta_z,
        )
        params_ser = simulator.parameters_series

        if "EMBP" in args.methods:
            # embp_kwargs = dict(
            #     outcome_type=args.outcome_type,
            #     ci=not args.no_ci,
            #     ci_method=args.ci_method,
            #     pbar=args.pbar,
            #     max_iter=args.max_iter,
            #     seed=seedi,
            #     n_bootstrap=args.n_bootstrap,
            #     gem=args.gem
            # )
            embp_model = EMBP(
                outcome_type=args.outcome_type,
                ci=not args.no_ci,
                ci_method=args.ci_method,
                pbar=args.pbar,
                max_iter=args.max_iter,
                seed=seedi,
                n_bootstrap=args.n_bootstrap,
                gem=args.gem,
                quasi_mc_K=args.quasi_K,
            )
        else:
            embp_model = None

        res_arrs = {}
        if args.ncores <= 1:
            for j in tqdm(range(args.nrepeat)):
                resi = trial_once_by_simulator_and_estimator(
                    type_outcome=args.outcome_type,
                    simulator=simulator,
                    estimator=embp_model,
                    seed=j + seedi,
                    methods=args.methods,
                )
                for k, arr in resi.items():
                    res_arrs.setdefault(k, []).append(arr)
        else:
            with mp.Pool(args.ncores) as pool:
                tmp_reses = [
                    pool.apply_async(
                        trial_once_by_simulator_and_estimator,
                        (
                            args.outcome_type,
                            simulator,
                            embp_model,
                            j + seedi,
                            args.methods,
                        ),
                    )
                    for j in range(args.nrepeat)
                ]
                for tmp_resi in tqdm(tmp_reses):
                    resi = tmp_resi.get()
                    for k, arr in resi.items():
                        res_arrs.setdefault(k, []).append(arr)

        # 3. collect results
        res_all = {
            "true": xr.DataArray(
                params_ser.values,
                dims=("params",),
                coords={"params": params_ser.index.values},
            ),
        }
        for k, arrs in res_arrs.items():
            if k == "EMBP":
                res_all[k] = xr.DataArray(
                    np.stack(arrs, axis=0),
                    dims=("repeat", "params", "statistic"),
                    coords={
                        "params": params_ser.index.values,
                        "statistic": embp_model.result_columns,
                    },
                )
            else:
                res_all[k] = xr.DataArray(
                    np.stack(arrs, axis=0)[:, None, :],
                    dims=("repeat", "params", "statistic"),
                    coords={
                        "params": ["beta_x"],
                        "statistic": ["estimate", "CI_1", "CI_2"],
                    },
                )

        res = xr.Dataset(res_all)

        # 4. print simple summary
        summary = {}
        for method in args.methods:
            summ_i = {}
            resi = res[method]
            diff = (
                resi.sel(params="beta_x", statistic="estimate").values
                - params_ser["beta_x"]
            )
            summ_i["bias"] = diff.mean()
            summ_i["mse"] = (diff**2).mean()
            if not args.no_ci:
                in_ci = (
                    resi.sel(params="beta_x", statistic="CI_1").values
                    <= params_ser["beta_x"]
                ) & (
                    resi.sel(params="beta_x", statistic="CI_2").values
                    >= params_ser["beta_x"]
                )
                summ_i["cov_rate"] = in_ci.mean()
            summary[method] = summ_i
        summary = pd.DataFrame.from_dict(summary)
        print(summary)
        summary = xr.DataArray(
            summary.values,
            dims=("metric", "method"),
            coords={
                "metric": summary.index.values,
                "method": summary.columns.values,
            },
        )
        res["summary"] = summary

        # 5. save results
        os.makedirs(args.root, exist_ok=True)
        ffn = os.path.join(
            args.root,
            f"{args.outcome_type if args.name is None else args.name}"
            f"-{datetime.now():%Y-%m-%d_%H-%M-%S}",
        )
        res.to_netcdf(ffn + ".nc")
        with open(ffn + ".json", "w") as f:
            json.dump(json_content, f)


if __name__ == "__main__":
    main()
