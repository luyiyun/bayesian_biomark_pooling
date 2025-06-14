import os
import os.path as osp
import multiprocessing as mp
import re
import json
import glob
from typing import Literal, Sequence
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import statsmodels.api as sm
import torch

from bayesian_biomarker_pooling.simulate import (
    BinarySimulator,
    ContinuousSimulator,
    default_serializer,
)
from bayesian_biomarker_pooling import EMBP
from bayesian_biomarker_pooling.utils import Timer


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


def method_naive(W, Y, Z, type_outcome: Literal["binary", "continue"]) -> np.ndarray:
    if Z is not None:
        W = np.concatenate([W[:, None], Z], axis=1)
    W = sm.add_constant(W)
    if type_outcome == "continue":
        model = sm.OLS(Y, W)
    else:
        model = sm.GLM(Y, W, sm.families.Binomial())
    res = model.fit()
    return np.r_[res.params[1], res.conf_int()[1, :]]


def analyze_data(
    X: np.ndarray,
    Y: np.ndarray,
    W: np.ndarray,
    S: np.ndarray | None,
    Z: np.ndarray | None,
    gpu: bool,
    ncores: int,
    outcome_type: Literal["binary", "continue"],
    methods: Sequence[Literal["embp", "xonly", "naive"]],
    embp_kwargs: dict,
) -> dict[str, np.ndarray]:
    if gpu and ncores > 1:
        torch.set_num_threads(1)
    res = {}
    for methodi in methods:
        with Timer() as t:
            if methodi == "xonly":
                resi = method_xonly(X, Y, Z, outcome_type)
            elif methodi == "naive":
                resi = method_naive(W, Y, Z, outcome_type)
            elif methodi == "embp":
                estimator = EMBP(
                    outcome_type=outcome_type,
                    **embp_kwargs,
                )
                estimator.fit(X, S, W, Y, Z)
                resi = estimator.params_
            else:
                raise ValueError(f"Unknown method: {methodi}")

        if methodi == "embp":
            resi = pd.concat(
                [resi, pd.DataFrame({"estimate": [t.interval]}, index=["time"])], axis=0
            )
        else:
            resi = pd.DataFrame.from_records(
                [resi, [t.interval, np.nan, np.nan]],
                index=["beta_x", "time"],
                columns=["estimate", "CI_1", "CI_2"],
            )
        res[methodi] = resi
    return res


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    # args["subcommand"]将会是子命令的名称

    # ============= 子命令：生成模拟数据 =============
    # region
    simu_parser = subparsers.add_parser("simulate", help="generate simulated data")
    simu_parser.add_argument(
        "-s",
        "--seed",
        default=0,
        type=int,
        help="random seed, default is 0",
    )
    simu_parser.add_argument(
        "-od",
        "--output_dir",
        default="./results/simulated_data",
        help=(
            "path to save simulated data (data.csv) and simulated "
            "parameters(params.json), default is./results/simulated_data"
        ),
    )
    simu_parser.add_argument(
        "-ot",
        "--outcome_type",
        default="continue",
        choices=["continue", "binary"],
        help="indicates the type of outcome, default is continue",
    )
    simu_parser.add_argument(
        "--n_studies",
        default=4,
        type=int,
        help="number of studies, default is 4",
    )
    simu_parser.add_argument(
        "--n_samples",
        default=100,
        type=int,
        nargs="+",
        help=(
            "number of samples per study, default is 100, "
            "can be a list whose length is n_studies"
        ),
    )
    simu_parser.add_argument(
        "--ratio_observed_x",
        default=0.1,
        type=float,
        nargs="+",
        help=(
            "ratio of observed X per study, default is 0.1, "
            "can be a list whose length is n_studies"
        ),
    )
    simu_parser.add_argument(
        "-bx",
        "--beta_x",
        default=0.0,
        type=float,
        help=("true beta_x, default is 0.0"),
    )
    simu_parser.add_argument(
        "--OR",
        default=None,
        type=float,
        help=(
            "true OR, default is None, only used when outcome_type is binary. "
            "When it's None, will use beta_x instead."
        ),
    )
    simu_parser.add_argument(
        "-b0",
        "--beta_0",
        default=(-0.5, -0.25, 0.25, 0.5),
        type=float,
        nargs="+",
        help=(
            "true beta_0, default is (-0.5, -0.25, 0.25, 0.5), "
            "can be a list whose length is n_studies, if set "
            "prevalence and outcome_type=binary, this option will be ignored"
        ),
    )
    simu_parser.add_argument(
        "-a",
        "--a",
        default=(-3, 1, -1, 3),
        type=float,
        nargs="+",
        help=(
            "true a, default is (-3, 1, -1, 3), can be a list whose length is n_studies"
        ),
    )
    simu_parser.add_argument(
        "-b",
        "--b",
        default=(0.5, 0.75, 1.25, 1.5),
        type=float,
        nargs="+",
        help=(
            "true b, default is (0.5, 0.75, 1.25, 1.5), "
            "can be a list whose length is n_studies"
        ),
    )
    simu_parser.add_argument(
        "-sx",
        "--sigma2_x",
        default=1.0,
        type=float,
        help="true sigma2_x, default is 1.0",
    )
    simu_parser.add_argument(
        "-se",
        "--sigma2_e",
        default=(1.0, 1.0, 1.0, 1.0),
        type=float,
        nargs="+",
        help=("true sigma2_e, default is 1.0, can be a list whose length is n_studies"),
    )
    simu_parser.add_argument(
        "-sy",
        "--sigma2_y",
        default=(0.5, 0.75, 1.0, 1.25),
        type=float,
        nargs="+",
        help=("true sigma2_y, default is 1.0, can be a list whose length is n_studies"),
    )
    simu_parser.add_argument(
        "-bz",
        "--beta_z",
        default=None,
        type=float,
        nargs="*",
        help=(
            "true beta_z, default is None, "
            "can be a list, the length of the list represents "
            "the number of covariates"
        ),
    )
    simu_parser.add_argument(
        "-pr",
        "--prevalence",
        default=None,
        type=float,
        help=(
            "prevalence, default is None, can be a float scalar. "
            "It is only used when outcome_type is binary."
        ),
    )
    simu_parser.add_argument(
        "-nr",
        "--n_repeats",
        default=1000,
        type=int,
        help="number of repeats, default is 1000",
    )
    # endregion

    # ============= 子命令：分析模拟数据 =============
    # region
    ana_parser = subparsers.add_parser("analyze", help="analyze simulated data")
    ana_parser.add_argument(
        "-ot",
        "--outcome_type",
        default="continue",
        choices=["continue", "binary"],
        help="indicates the type of outcome, default is continue",
    )
    ana_parser.add_argument(
        "-dd",
        "--data_dir",
        default="./results/simulated_data",
        help=("path to simulated data, default is ./results/simulated_data"),
    )
    ana_parser.add_argument(
        "-od",
        "--output_dir",
        default="./results/analyzed_results",
        help="path to save analyzed results, default is ./results/analyzed_results",
    )
    ana_parser.add_argument(
        "-m",
        "--methods",
        default=("embp", "xonly", "naive"),
        nargs="+",
        choices=("embp", "xonly", "naive"),
        help="methods to compare, default is embp, xonly, naive",
    )
    ana_parser.add_argument(
        "--no_ci",
        action="store_true",
        help="whether to use confidence interval, default is False (use CI)",
    )
    ana_parser.add_argument(
        "--ci_method",
        default="bootstrap",
        choices=("bootstrap", "sem"),
        help="method to estimate CI, default is bootstrap",
    )
    ana_parser.add_argument(
        "--max_iter",
        default=None,
        type=int,
        help="maximum number of iterations, default is None "
        "(500 for continue, 300 for binary)",
    )
    ana_parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="random seed, default is 0",
    )
    ana_parser.add_argument(
        "--n_bootstrap",
        default=200,
        type=int,
        help="number of bootstrap samples for CI estimation, default is 200",
    )
    ana_parser.add_argument(
        "--gem",
        action="store_true",
        help="whether to use generalized EM for binary outcome (M step "
        "only update one Newton step), default is False (use traditional EM)",
    )
    ana_parser.add_argument(
        "-qK",
        "--quasi_K",
        default=100,
        type=int,
        help="number of quasi-MC samples for binary outcome, default is 100",
    )
    ana_parser.add_argument(
        "-bs",
        "--binary_solve",
        default="lap",
        choices=["lap", "is"],
        help="method to aproximate posterior distribution for binary "
        "outcome, can be lap(Laplace approximation) or "
        "is(importance sampling), default is lap",
    )
    ana_parser.add_argument(
        "-ismk",
        "--importance_sampling_maxK",
        default=5000,
        type=int,
        help="maximum number of samples for importance sampling, default is 5000",
    )
    ana_parser.add_argument(
        "-g",
        "--gpu",
        action="store_true",
        help="whether to use GPU, default is False",
    )
    ana_parser.add_argument(
        "--delta2",
        default=None,
        type=float,
        help="delta2 for stop criterion, default is None "
        "(1e-5 for continue, 1e-2 for binary)",
    )
    ana_parser.add_argument(
        "-epb",
        "--embp_progress_bar",
        action="store_true",
        help="whether to show progress bar when run embp, default is False",
    )
    ana_parser.add_argument(
        "-nc",
        "--ncores",
        default=1,
        type=int,
        help="number of cores to use, default is 1",
    )
    # endregion

    # ============= 子命令：计算评价指标 =============
    # region
    eval_parser = subparsers.add_parser(
        "evaluate", help="evaluate the performance of methods"
    )
    eval_parser.add_argument(
        "-ad",
        "--analyzed_dir",
        default="./results/analyzed_results",
        help="path to save evaluated results, default is ./results/evaluated_results",
    )
    eval_parser.add_argument(
        "-of",
        "--output_file",
        default="evaluated_results.csv",
        help="path to save evaluated results, default is evaluated_results.csv",
    )
    # endregion

    # ============= 子命令：总结评价指标 =============
    # region
    summ_parser = subparsers.add_parser(
        "summarize", help="summarize the performance of methods"
    )
    summ_parser.add_argument(
        "-efp",
        "--evaluated_file_pattern",
        default="./results/evaluated_results/*.csv",
        help="file pattern to evaluated results, default is ./results/evaluated_results/*.csv",
    )
    summ_parser.add_argument(
        "-of",
        "--output_file",
        default="summarized_results.xlsx",
        help="path to save evaluated results, default is summarized_results.xlsx, must be a xlsx file",
    )
    summ_parser.add_argument(
        "-sp",
        "--summarize_parameters",
        type=str,
        nargs="+",
        default=("beta_x", "ratio_observed_x"),
        help="parameters to summarize, default is beta_x, ratio_observed_x",
    )

    args = parser.parse_args()

    # ================= 读取模拟数据，进行模拟实验 =================
    fn = "./example_binary_error/data.csv"
    df_all = pd.read_csv(fn, index_col=None)
    for ri in range(760, 770):
        print(f"repeat {ri}")
        df = df_all.query(f"repeat == {ri}")
        print(df.head())
        embp_kwargs = {
            "ci": not args.no_ci,
            "ci_method": args.ci_method,
            "pbar": True,
            "max_iter": args.max_iter,
            "seed": args.seed,
            "n_bootstrap": args.n_bootstrap,
            "gem": args.gem,
            "quasi_mc_K": args.quasi_K,
            "delta2": args.delta2,
            "binary_solve": "vi",
            "device": "cuda:0" if args.gpu else "cpu",
            "importance_sampling_maxK": args.importance_sampling_maxK,
        }

        zind = df.columns.map(lambda x: re.search(r"Z\d*", x) is not None)
        X = df["X"].values
        Y = df["Y"].values
        W = df["W"].values
        S = df["S"].values
        Z = df.loc[:, zind].values if zind.any() else None

        res = analyze_data(
            X,
            Y,
            W,
            S,
            Z,
            args.gpu,
            args.ncores,
            args.outcome_type,
            args.methods,
            embp_kwargs,
        )

        for methodi, resi in res.items():
            print(f"method: {methodi}")
            print(resi)
        print()
    return

    # ================= 读取实验结果和模拟参数，计算评价指标 =================
    if args.subcommand == "evaluate":
        output_file = osp.join(args.analyzed_dir, args.output_file)
        if osp.exists(output_file):
            raise ValueError(
                f"Output file {output_file} already exists, please remove it first."
            )

        with open(osp.join(args.analyzed_dir, "params.json"), "r") as f:
            ana_args = json.load(f)
        with open(osp.join(ana_args["data_dir"], "params.json"), "r") as f:
            simu_args = json.load(f)
        true_beta_x = simu_args["betax"]

        res = xr.load_dataset(osp.join(args.analyzed_dir, "analyzed_results.nc"))
        index, res_df = [], defaultdict(list)
        for k, da in res.items():
            index.append(k)
            diff = da.sel(params="beta_x", statistic="estimate").values - true_beta_x
            res_df["bias"].append(diff.mean())
            res_df["mse"].append((diff**2).mean())
            res_df["bias_se"].append(diff.std() / np.sqrt(diff.shape[0]))
            if not ana_args["no_ci"]:
                in_ci = (
                    da.sel(params="beta_x", statistic="CI_1").values <= true_beta_x
                ) & (da.sel(params="beta_x", statistic="CI_2").values >= true_beta_x)
                res_df["cov_rate"].append(in_ci.mean())
            time_arr = da.sel(params="time", statistic="estimate").values
            res_df["time_mean"].append(time_arr.mean())
            res_df["time_se"].append(time_arr.std() / np.sqrt(time_arr.shape[0]))

        res_df = pd.DataFrame(res_df, index=index)
        print(res_df)
        res_df.to_csv(output_file)

    # ================= 读取读取多次实验的结果，整合为一个表格 =================
    if args.subcommand == "summarize":
        all_res = []
        for fn in glob.glob(args.evaluated_file_pattern):
            path = osp.dirname(fn)
            with open(osp.join(path, "params.json"), "r") as f:
                ana_args = json.load(f)
            with open(osp.join(ana_args["data_dir"], "params.json"), "r") as f:
                simu_args = json.load(f)
            summ_df = pd.read_csv(fn, index_col=0)
            for k in args.summarize_parameters:
                if k not in simu_args:
                    raise ValueError(f"Parameter {k} not found in simu_args.")
                v = simu_args[k]
                if isinstance(v, (list, tuple)):
                    if len(v) == 1 or all(vi == v[0] for vi in v[1:]):
                        v = v[0]
                    else:
                        v = "-".join(map(str, v))
                summ_df[k] = v
            all_res.append(summ_df)
        all_res = pd.concat(all_res, axis=0)

        # all_res["time_sd"] *= np.sqrt(1000)*100
        all_res["time_se"] *= 100
        all_res["bias"] *= 100
        all_res["mse"] *= 100
        all_res["cov_rate"] *= 100
        # all_res["bias_sd"] *= np.sqrt(1000) *100
        all_res["bias_se"] *= 100

        all_res["time"] = [
            # f"{m:.4f}±{s:.4f}" for m, s in zip(all_res["time_mean"], all_res["time_sd"])
            f"{m:.4f}±{s:.4f}"
            for m, s in zip(all_res["time_mean"], all_res["time_se"])
        ]
        all_res["bias"] = [
            # f"{m:.4f}({s:.4f})" for m, s in zip(all_res["bias"], all_res["bias_sd"])
            f"{m:.4f}({s:.4f})"
            for m, s in zip(all_res["bias"], all_res["bias_se"])
        ]

        # all_res.drop(columns=["time_mean", "time_sd", "bias_sd"], inplace=True)
        all_res.drop(columns=["time_mean", "time_se", "bias_se"], inplace=True)
        all_res.index.name = "methods"
        all_res.set_index(args.summarize_parameters, append=True, inplace=True)
        all_res = all_res.unstack(level=-1).swaplevel(0, -1).swaplevel(0, -1, axis=1)
        all_res.sort_index(axis=0, inplace=True)
        all_res.sort_index(axis=1, inplace=True)
        print(all_res.to_string())
        with pd.ExcelWriter(args.output_file) as writer:
            all_res.to_excel(writer)


if __name__ == "__main__":
    main()
