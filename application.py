import logging
import os
import os.path as osp
from argparse import ArgumentParser

import pandas as pd

from src.method import SimpleModel, HierachicalModel, HierachicalModel_WX

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


def main():
    parser = ArgumentParser()
    parser.add_argument("--save_root", type=str, default="./results/")
    parser.add_argument(
        "--save_action", choices=["cover", "raise", "ignore"], default="raise"
    )

    parser.add_argument("--ncores", type=int, default=1)
    parser.add_argument(
        "--model",
        type=str,
        choices=["simple", "hier", "hier_wx"],
        default="hier",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["pymc", "blackjax", "numpyro", "nutpie", "vi"],
        default="pymc",
    )

    # bayesian inference settings
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
        "--prior_betax",
        type=str,
        choices=["flat", "normal"],
        default="flat",
    )
    parser.add_argument("--prior_a_std", type=float, default=1.0)
    parser.add_argument("--prior_b_std", type=float, default=1.0)
    parser.add_argument("--prior_beta0_std", type=float, default=1.0)
    # parser.add_argument("--use_hier_x_prior", action="store_true")
    parser.add_argument("--direct_x_prior", action="store_true")
    parser.add_argument(
        "--direction", type=str, choices=["w->x", "x->w"], default="x->w"
    )
    args = parser.parse_args()

    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)

    # read data
    dat = pd.read_csv(osp.join(save_root, "BRCA1_simu.csv"), index_col=0)

    # fit model
    analysis_kwargs = dict(
        solver=args.solver,
        prior_sigma_ws=args.prior_sigma_ws,
        prior_sigma_ab0=args.prior_sigma_ab0,
        prior_betax=args.prior_betax,
        prior_a_std=args.prior_a_std,
        prior_b_std=args.prior_b_std,
        prior_beta0_std=args.prior_beta0_std,
        hier_prior_on_x=not args.direct_x_prior,
        pbar=True,
        nchains=4,
        ntunes=10000,
        nsample=10000,
    )
    if args.model == "simple":
        model = SimpleModel(**analysis_kwargs)
    elif args.model == "hier":
        model = HierachicalModel(**analysis_kwargs)
    elif args.model == "hier_wx":
        model = HierachicalModel_WX(**analysis_kwargs)
    else:
        raise NotImplementedError

    model.fit(dat)
    baye_res = model.summary()
    print(baye_res)
    baye_res.to_csv(osp.join(save_root, "BRCA1_res.csv"))


if __name__ == "__main__":
    main()
