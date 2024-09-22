import logging
import os
import os.path as osp
import json
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from bayesian_biomarker_pooling.simulate import Simulator


logger = logging.getLogger("experiment.BBP.simulate_data")


def main():
    logger_simu = logging.getLogger("main.simulate")
    logger_simu.setLevel(logging.WARNING)
    logging.basicConfig(
        format="[%(asctime)s][%(name)s][%(levelname)s]: %(message)s",
        level=logging.INFO,
    )

    parser = ArgumentParser()
    parser.add_argument("--save_root", type=str, default="./data/")
    parser.add_argument("--save_prefix", type=str, default="")
    parser.add_argument("--nrepeat", type=int, default=10)
    parser.add_argument("--prevalence", type=float, default=0.50)
    parser.add_argument("--beta0", type=float, nargs="+", default=[0.0])
    parser.add_argument("--betax", type=float, default=1.0)
    parser.add_argument("--betaz", type=float, nargs="+", default=None)
    parser.add_argument("--OR", type=float, default=None)
    parser.add_argument("--sigma2x", type=float, default=1.0)
    parser.add_argument("--sigma2e", type=float, nargs="+", default=[1.0])
    parser.add_argument("--a", type=float, nargs="+", default=[-3, 1, -1, 3])
    parser.add_argument(
        "--b", type=float, nargs="+", default=[0.5, 0.75, 1.25, 1.5]
    )
    parser.add_argument("--n_studies", type=int, default=4)
    parser.add_argument("--a_mu", type=float, default=0.0)
    parser.add_argument("--a_sigma", type=float, default=3.0)
    parser.add_argument("--b_mu", type=float, default=0.0)
    parser.add_argument("--b_sigma", type=float, default=3.0)
    parser.add_argument("--nSamples", type=int, nargs="+", default=[100])
    parser.add_argument("--nKnowX", type=int, nargs="+", default=[20])
    parser.add_argument(
        "--n_knowX_balance",
        action="store_true",
        help=(
            "if set n_knowX_balance, "
            "the y of samples which have X will be balanced."
        ),
    )
    parser.add_argument(
        "--direction", type=str, choices=["w->x", "x->w"], default="x->w"
    )
    parser.add_argument(
        "--type_outcome",
        type=str,
        choices=["continue", "binary", "survival"],
        default="binary",
    )
    parser.add_argument("--censor_rate", type=float, default=0.2)

    args = parser.parse_args()

    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)

    if args.OR is not None and args.betax is not None:
        logger.info(
            "OR and betax are set simultaneously, "
            f"use OR={args.OR}, betax={np.log(args.OR)}"
        )
    elif args.OR is not None:
        args.betax = np.log(args.OR)

    # 这些参数的长度需要保持一致或者是scalar
    if args.n_studies is None:
        args.n_studies = None
        for arg_i in [args.a, args.b, args.sigma2e]:
            if len(arg_i) != 1:
                if args.n_studies is None:
                    args.n_studies = len(arg_i)
                else:
                    assert len(arg_i) == args.n_studies

    simulator = Simulator(
        beta_x=args.betax,
        beta_z=args.betaz,
        beta_0=args.beta0[0] if len(args.beta0) == 1 else args.beta0,
        prevalence=args.prevalence,
        a=args.a,
        b=args.b,
        sigma2_e=args.sigma2e[0] if len(args.sigma2e) == 1 else args.sigma2e,
        sigma2_x=args.sigma2x,
        n_sample_per_studies=(
            args.nSamples[0] if len(args.nSamples) == 1 else args.nSamples
        ),
        n_knowX_per_studies=(
            args.nKnowX[0] if len(args.nKnowX) == 1 else args.nKnowX
        ),
        n_knowX_balance=args.n_knowX_balance,
        direction=args.direction,
        type_outcome=args.type_outcome,
        censor_rate=args.censor_rate,
    )

    # save name
    time_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_name = osp.join(
        args.save_root,
        (
            f"{args.save_prefix}_{time_name}"
            if args.save_prefix
            else time_name
        ),
    )
    logger.info(f"save into {save_name}")

    with pd.HDFStore(save_name + ".h5") as store:
        for i in tqdm(range(args.nrepeat)):
            sim_dat = simulator.simulate(i)
            sim_dat.to_hdf(store, key=f"s{i}")

    parameters = {}
    for k, v in simulator.parameters.items():
        if isinstance(v, np.ndarray):
            parameters[k] = v.tolist()
        elif isinstance(v, np.integer):
            parameters[k] = int(v)
        elif isinstance(v, np.floating):
            parameters[k] = float(v)
        else:
            parameters[k] = v

    with open(save_name + ".json", "w") as f:
        json.dump(parameters, f)


if __name__ == "__main__":
    main()
