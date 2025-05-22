import os
import os.path as osp
import multiprocessing as mp
import re
import json
from typing import Literal, Sequence
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import statsmodels.api as sm
import torch

from bayesian_biomarker_pooling.simulate import BinarySimulator, ContinuousSimulator
from bayesian_biomarker_pooling import EMBP


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
        "-se",
        "--sigma2_e",
        default=(0.5, 0.75, 1.0, 1.25),
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
        nargs="+",
        help=(
            "prevalence, default is None, can be a list whose length "
            "is n_studies. It is only used when outcome_type is binary."
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

    args = parser.parse_args()

    # ================= 模拟数据，并保存 =================
    if args.subcommand == "simulate":
        if osp.exists(args.output_dir):
            raise ValueError(
                f"Output directory {args.output_dir} already "
                "exists, please remove it first."
            )

        def proc_args(x):
            if x is None:
                return x
            if isinstance(x, (list, tuple)) and len(x) == 1:
                return [x[0]] * args.n_studies
            assert len(x) == args.n_studies, (
                f"length of {x} must be equal to n_studies {args.n_studies}"
            )
            return x

        ratio_observed_x = proc_args(args.ratio_observed_x)
        n_sample_per_study = proc_args(args.n_samples)
        n_knowX_per_study = [
            int(r * n) for r, n in zip(ratio_observed_x, n_sample_per_study)
        ]

        if args.outcome_type == "binary":
            simulator = BinarySimulator(
                beta0=proc_args(args.beta_0),
                a=proc_args(args.a),
                b=proc_args(args.b),
                sigma2_e=proc_args(args.sigma2_e),
                n_sample_per_studies=n_sample_per_study,
                n_knowX_per_studies=n_knowX_per_study,
                betaz=args.beta_z,
                OR=args.OR or np.exp(args.beta_x),
                prevalence=proc_args(args.prevalence),
                n_knowX_balance=True,
            )
        else:
            simulator = ContinuousSimulator(
                beta0=proc_args(args.beta_0),
                a=proc_args(args.a),
                b=proc_args(args.b),
                sigma2_e=proc_args(args.sigma2_e),
                n_sample_per_studies=n_sample_per_study,
                n_knowX_per_studies=n_knowX_per_study,
                betaz=args.beta_z,
                betax=args.beta_x,
                sigma2_y=proc_args(args.sigma2_y),
            )

        df_all = []
        for i in tqdm(range(args.n_repeats), desc="Simulate: "):
            df = simulator.simulate(seed=i)
            df["repeat"] = i
            df_all.append(df)
        df_all = pd.concat(df_all, ignore_index=True)

        os.makedirs(args.output_dir, exist_ok=False)  # 确保目录不存在
        df_all.to_csv(osp.join(args.output_dir, "data.csv"), index=False)
        simulator.save(osp.join(args.output_dir, "params.json"))

        return

    # ================= 读取模拟数据，进行模拟实验 =================
    if args.subcommand == "analyze":
        if osp.exists(args.output_dir):
            raise ValueError(
                f"Output directory {args.output_dir} already "
                "exists, please remove it first."
            )

        fn = osp.join(args.data_dir, "data.csv")
        df = pd.read_csv(fn, index_col=None)

        # with open(osp.join(args.data_dir, "params.json"), "r") as f:
        #     simu_args = json.load(f)

        if "repeat" not in df.columns:
            df_iter = [(0, df)]
        else:
            df_iter = df.groupby("repeat")

        embp_kwargs = {
            "ci": not args.no_ci,
            "ci_method": args.ci_method,
            "pbar": args.embp_progress_bar,
            "max_iter": args.max_iter,
            "seed": args.seed,
            "n_bootstrap": args.n_bootstrap,
            "gem": args.gem,
            "quasi_mc_K": args.quasi_K,
            "delta2": args.delta2,
            "binary_solve": args.binary_solve,
            "device": "cuda:0" if args.gpu else "cpu",
            "importance_sampling_maxK": args.importance_sampling_maxK,
        }
        res_all = {k: [] for k in args.methods}

        if args.ncores <= 1:
            for i, dfi in tqdm(df_iter, desc="Analyze: "):
                zind = dfi.columns.map(lambda x: re.search(r"Z\d*", x) is not None)
                X = dfi["X"].values
                Y = dfi["Y"].values
                W = dfi["W"].values
                S = dfi["S"].values
                Z = dfi.loc[:, zind].values if zind.any() else None

                resi = analyze_data(
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

                for k, v in resi.items():
                    res_all[k].append(v)

                # if i >= 5:
                #     break

        elif args.gpu:
            pass
            raise NotImplementedError("GPU multi-processing is not implemented yet.")
            # n_cudas = torch.cuda.device_count()
            # if n_cudas != args.ncores:
            #     print(
            #         f"Only {n_cudas} gpus, thus "
            #         f"open {n_cudas} subprocesses, not {args.ncores}."
            #     )

            # manager = mp_torch.Manager()
            # q = manager.Queue()
            # for i in range(n_cudas):
            #     q.put(f"cuda:{i}")

            # with mp_torch.Pool(n_cudas) as pool:
            #     tmp_reses = [
            #         pool.apply_async(
            #             trial_once_by_simulator_and_estimator,
            #             kwds={
            #                 "type_outcome": args.outcome_type,
            #                 "simulator": simulator,
            #                 "estimator": embp_kwargs,
            #                 "seed": j + seedi,
            #                 "methods": args.methods,
            #                 "gpu_and_mp": False,
            #                 "logging": False,
            #                 "queue": q,
            #             },
            #         )
            #         for j in range(args.nrepeat)
            #     ]
            #     for tmp_resi in tqdm(tmp_reses):
            #         resi = tmp_resi.get()
            #         for k, arr in resi.items():
            #             res_arrs.setdefault(k, []).append(arr)
        else:  # use cpu multi-processing
            with mp.Pool(args.ncores) as pool:
                tmp_reses = []
                for i, dfi in df_iter:
                    zind = dfi.columns.map(lambda x: re.search(r"Z\d*", x) is not None)
                    X = dfi["X"].values
                    Y = dfi["Y"].values
                    W = dfi["W"].values
                    S = dfi["S"].values
                    Z = dfi.loc[:, zind].values if zind.any() else None
                    tmp_resi = pool.apply_async(
                        analyze_data,
                        (
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
                        ),
                    )
                    tmp_reses.append(tmp_resi)
                for tmp_resi in tqdm(tmp_reses):
                    resi = tmp_resi.get()
                    for k, v in resi.items():
                        res_all[k].append(v)

        res_all = {
            k: xr.DataArray(
                np.stack([vi.values for vi in v], axis=0),
                dims=("repeat", "params", "statistic"),
                coords={
                    "params": v[0].index.values,
                    "statistic": v[0].columns.values,
                },
            )
            if k == "embp"
            else xr.DataArray(
                np.stack(v, axis=0)[:, None, :],
                dims=("repeat", "params", "statistic"),
                coords={
                    "params": ["beta_x"],
                    "statistic": ["estimate", "CI_1", "CI_2"],
                },
            )
            for k, v in res_all.items()
        }
        res_all = xr.Dataset(res_all)

        os.makedirs(args.output_dir, exist_ok=False)  # 确保目录不存在
        res_all.to_netcdf(osp.join(args.output_dir, "analyzed_results.nc"))
        with open(osp.join(args.output_dir, "params.json"), "w") as f:
            ana_args = args.__dict__
            json.dump(ana_args, f)

        return

    # ================= 读取实验结果和模拟参数，计算评价指标 =================
    if args.subcommand == "evaluate":
        if osp.exists(args.output_file):
            raise ValueError(
                f"Output file {args.output_file} already "
                "exists, please remove it first."
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
            if not ana_args["no_ci"]:
                in_ci = (
                    da.sel(params="beta_x", statistic="CI_1").values <= true_beta_x
                ) & (da.sel(params="beta_x", statistic="CI_2").values >= true_beta_x)
                res_df["cov_rate"].append(in_ci.mean())

        res_df = pd.DataFrame(res_df, index=index)
        print(res_df)
        res_df.to_csv(args.output_file)

    # if args.gpu:
    #     mp_torch.set_start_method("spawn")

    # log_level = {
    #     "error": logging.ERROR,
    #     "warn": logging.WARNING,
    #     "info": logging.INFO,
    #     "debug": logging.DEBUG,
    # }[args.log]
    # logger = logging.getLogger("EMBP")
    # logger.setLevel(log_level)
    # for handler in logger.handlers:
    #     if isinstance(handler, logging.StreamHandler):
    #         handler.setLevel(log_level)

    # if args.test:
    #     if args.outcome_type == "continue":
    #         temp_test_continue(ci=not args.no_ci, ve_method=args.ci_method)
    #     elif args.outcome_type == "binary":
    #         temp_test_binary(
    #             ci=not args.no_ci,
    #             seed=1,
    #             nsample=100,
    #             n_knowX=10,
    #             beta_x=args.beta_x[0],
    #             binary_solve=args.binary_solve,
    #             gpu=args.gpu,
    #             ci_method=args.ci_method,
    #         )
    #     return

    # # 模拟实验：
    # # 1. 不同样本量，不同缺失比例下的效果,
    # # 2. 一类错误 & 效能
    # # 3. 把参数默认值搞清楚

    # # if args.skip_dup:
    # #     runned_configs = []
    # #     for fn in os.listdir(args.root):
    # #         if fn.startswith(
    # #             args.outcome_type if args.name is None else args.name
    # #         ) and fn.endswith(".json"):
    # #             with open(osp.join(args.root, fn), "r") as f:
    # #                 runned_configs.append(json.load(f))
    # if args.skip is not None:
    #     skip_set = [
    #         tuple([float(s) for s in skip_str.split(",")])
    #         for skip_str in args.skip
    #     ]
    # else:
    #     skip_set = []

    # for i, (ns, rx, bx) in enumerate(
    #     product(
    #         args.nsample_per_studies, args.ratio_x_per_studies, args.beta_x
    #     )
    # ):
    #     print(
    #         f"nsample per studies: {ns}, "
    #         f"ratio of observed x: {rx:.2f}, true beta x: {bx:.2f}"
    #     )
    #     if (ns, rx, bx) in skip_set:
    #         print("skip")
    #         continue

    #     seedi = args.seed + i
    #     nx = int(rx * ns)

    #     json_content = deepcopy(args.__dict__)
    #     json_content["nsample"] = ns
    #     json_content["ratiox"] = rx
    #     json_content["betax"] = bx
    #     json_content["seed"] = seedi
    #     json_content["nx"] = nx

    #     simulator = Simulator(
    #         type_outcome=args.outcome_type,
    #         beta_x=bx,
    #         sigma2_y=[0.5, 0.75, 1.0, 1.25],
    #         sigma2_e=[0.5, 0.75, 1.0, 1.25],
    #         beta_0=(
    #             None
    #             if args.outcome_type == "binary"
    #             and args.prevalence is not None
    #             else args.beta_0
    #         ),
    #         prevalence=args.prevalence,
    #         n_sample_per_studies=ns,
    #         n_knowX_per_studies=nx,
    #         beta_z=args.beta_z,
    #     )
    #     params_ser = simulator.parameters_series

    #     if "EMBP" in args.methods:
    #         embp_kwargs = dict(
    #             outcome_type=args.outcome_type,
    #             ci=not args.no_ci,
    #             ci_method=args.ci_method,
    #             pbar=args.pbar,
    #             max_iter=args.max_iter,
    #             seed=seedi,
    #             n_bootstrap=args.n_bootstrap,
    #             gem=args.gem,
    #             quasi_mc_K=args.quasi_K,
    #             delta2=args.delta2,
    #             binary_solve=args.binary_solve,
    #             device="cuda:0" if args.gpu else "cpu",
    #             importance_sampling_maxK=args.importance_sampling_maxK,
    #         )
    #         embp_model = EMBP(**embp_kwargs)
    #     else:
    #         embp_kwargs = None

    #     res_arrs = {}
    #     if args.ncores <= 1:
    #         for j in tqdm(range(args.nrepeat)):
    #             resi = trial_once_by_simulator_and_estimator(
    #                 type_outcome=args.outcome_type,
    #                 simulator=simulator,
    #                 estimator=embp_kwargs,
    #                 seed=j + seedi,
    #                 methods=args.methods,
    #             )
    #             for k, arr in resi.items():
    #                 res_arrs.setdefault(k, []).append(arr)
    #     else:
    #         if args.gpu:
    #             n_cudas = torch.cuda.device_count()
    #             if n_cudas != args.ncores:
    #                 print(
    #                     f"Only {n_cudas} gpus, thus "
    #                     f"open {n_cudas} subprocesses, not {args.ncores}."
    #                 )

    #             manager = mp_torch.Manager()
    #             q = manager.Queue()
    #             for i in range(n_cudas):
    #                 q.put(f"cuda:{i}")

    #             with mp_torch.Pool(n_cudas) as pool:
    #                 tmp_reses = [
    #                     pool.apply_async(
    #                         trial_once_by_simulator_and_estimator,
    #                         kwds={
    #                             "type_outcome": args.outcome_type,
    #                             "simulator": simulator,
    #                             "estimator": embp_kwargs,
    #                             "seed": j + seedi,
    #                             "methods": args.methods,
    #                             "gpu_and_mp": False,
    #                             "logging": False,
    #                             "queue": q,
    #                         },
    #                     )
    #                     for j in range(args.nrepeat)
    #                 ]
    #                 for tmp_resi in tqdm(tmp_reses):
    #                     resi = tmp_resi.get()
    #                     for k, arr in resi.items():
    #                         res_arrs.setdefault(k, []).append(arr)
    #         else:
    #             with mp.Pool(args.ncores) as pool:
    #                 tmp_reses = [
    #                     pool.apply_async(
    #                         trial_once_by_simulator_and_estimator,
    #                         (
    #                             args.outcome_type,
    #                             simulator,
    #                             embp_kwargs,
    #                             j + seedi,
    #                             args.methods,
    #                         ),
    #                     )
    #                     for j in range(args.nrepeat)
    #                 ]
    #                 for tmp_resi in tqdm(tmp_reses):
    #                     resi = tmp_resi.get()
    #                     for k, arr in resi.items():
    #                         res_arrs.setdefault(k, []).append(arr)

    #     # 3. collect results
    #     res_all = {
    #         "true": xr.DataArray(
    #             params_ser.values,
    #             dims=("params",),
    #             coords={"params": params_ser.index.values},
    #         ),
    #     }
    #     for k, arrs in res_arrs.items():
    #         if k == "EMBP":
    #             res_all[k] = xr.DataArray(
    #                 np.stack(arrs, axis=0),
    #                 dims=("repeat", "params", "statistic"),
    #                 coords={
    #                     "params": params_ser.index.values,
    #                     "statistic": embp_model.result_columns,
    #                 },
    #             )
    #         else:
    #             res_all[k] = xr.DataArray(
    #                 np.stack(arrs, axis=0)[:, None, :],
    #                 dims=("repeat", "params", "statistic"),
    #                 coords={
    #                     "params": ["beta_x"],
    #                     "statistic": ["estimate", "CI_1", "CI_2"],
    #                 },
    #             )

    #     res = xr.Dataset(res_all)

    #     # 4. print simple summary
    #     summary = {}
    #     for method in args.methods:
    #         summ_i = {}
    #         resi = res[method]
    #         diff = (
    #             resi.sel(params="beta_x", statistic="estimate").values
    #             - params_ser["beta_x"]
    #         )
    #         summ_i["bias"] = diff.mean()
    #         summ_i["mse"] = (diff**2).mean()
    #         if not args.no_ci:
    #             in_ci = (
    #                 resi.sel(params="beta_x", statistic="CI_1").values
    #                 <= params_ser["beta_x"]
    #             ) & (
    #                 resi.sel(params="beta_x", statistic="CI_2").values
    #                 >= params_ser["beta_x"]
    #             )
    #             summ_i["cov_rate"] = in_ci.mean()
    #         summary[method] = summ_i
    #     summary = pd.DataFrame.from_dict(summary)
    #     print(summary)
    #     summary = xr.DataArray(
    #         summary.values,
    #         dims=("metric", "method"),
    #         coords={
    #             "metric": summary.index.values,
    #             "method": summary.columns.values,
    #         },
    #     )
    #     res["summary"] = summary

    #     # 5. save results
    #     os.makedirs(args.root, exist_ok=True)
    #     ffn = os.path.join(
    #         args.root,
    #         f"{args.outcome_type if args.name is None else args.name}"
    #         f"-{datetime.now():%Y-%m-%d_%H-%M-%S}",
    #     )
    #     res.to_netcdf(ffn + ".nc")
    #     with open(ffn + ".json", "w") as f:
    #         json.dump(json_content, f)


if __name__ == "__main__":
    main()
