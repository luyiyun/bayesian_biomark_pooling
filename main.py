import logging
import multiprocessing as mp
import os
import os.path as osp
import re
import shutil
from argparse import ArgumentParser
from itertools import product
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluate import evaluate
from src.method import Model
from src.simulate import Simulator2

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
    simulator: Simulator2,
    analysis_kwargs: Dict = {},
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sim_dat = simulator.simulate(seed)
    sim_dat["S"] = sim_dat["S"] - 1  # 模拟实验是从1开始的，现在改成0

    model = Model(seed=seed, **analysis_kwargs)
    model.fit(sim_dat)
    baye_res = model.summary()

    eval_res = evaluate(simulator.parameters, baye_res)
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
        ndraws: int = 1000,
        ntunes: int = 1000,
        ncores: int = 1,
        direction: Literal["x->w", "w->x"] = "x->w",
        solver: Literal[
            "pymc", "blackjax", "numpyro", "nutpie", "vi"
        ] = "pymc",
        pytensor_cache: Optional[str] = None,
        prevalence: float = 0.05,
        OR: float = 1.25,
        sigma2_e: Union[float, Sequence[float]] = 0.1,
        sigma2_x: Union[float, Sequence[float]] = 1.0,
        a: Union[float, Sequence[float]] = [-3, 1, -1, 3],
        b: Union[float, Sequence[float]] = [0.5, 0.75, 1.25, 1.5],
        prior_sigma_ws: Literal["gamma", "inv_gamma"] = "gamma",
        prior_sigma_ab0: Literal["half_cauchy", "half_flat"] = "half_cauchy",
        n_sample_per_studies: Union[int, Sequence[int]] = 1000,
        n_knowX_per_studies: Union[int, Sequence[int]] = 100,
        use_hier_x_prior: bool = True,
    ) -> None:
        self._nrepeat = nrepeat
        self._ncores = ncores
        self._simulator = Simulator2(
            prevalence=prevalence,
            OR=OR,
            direction=direction,
            sigma2_e=sigma2_e,
            sigma2_x=sigma2_x,
            a=a,
            b=b,
            n_sample_per_studies=n_sample_per_studies,
            n_knowX_per_studies=n_knowX_per_studies,
        )
        self._analysis_kwargs = dict(
            solver=solver,
            prior_sigma_ws=prior_sigma_ws,
            prior_sigma_ab0=prior_sigma_ab0,
            nsample=ndraws,
            ntunes=ntunes,
            hier_prior_on_x=use_hier_x_prior,
        )
        self._simul_name = self._simulator.name
        self._trial_name = (
            (
                "prev%.2f-OR%.2f-direct@%s-sigma2x%.1f"
                "-priorSigmaWs@%s-priorSigmaAB0@%s"
            )
            % (
                prevalence,
                OR,
                direction,
                # sigma2_e,
                sigma2_x,
                prior_sigma_ws,
                prior_sigma_ab0,
            )
        ).replace(".", "_")
        self._name = self._trial_name
        self._pytensor_cache = pytensor_cache

    def simulate(
        self, nrepeat: Optional[int] = None, ncores: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        nrepeat = nrepeat if nrepeat is not None else self._nrepeat
        ncores = ncores if ncores is not None else self._ncores
        arr = []
        if ncores <= 1:
            for i in tqdm(
                range(nrepeat), desc="Simulate %s: " % self._simul_name
            ):
                sim_dat = self._simulator.simulate(i)
                arr.append(sim_dat.values.astype(float))
        else:
            with mp.Pool(ncores) as pool:
                temp_reses = [
                    pool.apply_async(self._simulator.simulate, (seedi,))
                    for seedi in range(nrepeat)
                ]
                for temp_resi in tqdm(
                    temp_reses, desc="Simulate %s: " % self._simul_name
                ):
                    sim_dat = temp_resi.get()
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
        logger_main.info("Task name: %s" % self._trial_name)
        if ncores <= 1:
            for i in tqdm(range(nrepeat), desc="Pipeline: "):
                (
                    (sim_i, sim_col),
                    (ana_i, ana_col, ana_ind),
                    (eva_i, eva_col, eva_ind),
                ) = _pipeline(i, self._simulator, self._analysis_kwargs)
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
                        _pipeline,
                        (seedi, self._simulator, self._analysis_kwargs),
                    )
                    for seedi in range(nrepeat)
                ]
                for temp_resi in tqdm(temp_reses, desc="Pipeline: "):
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
    arr: np.ndarray,
    index: Sequence[str],
    columns: Sequence[str],
) -> pd.DataFrame:
    # arr_mean, arr_std = arr.mean(axis=0), arr.std(axis=0)
    # res = np.full_like(arr_mean, "", dtype="U25")
    # for i in range(arr_mean.shape[0]):
    #     for j in range(arr_mean.shape[1]):
    #         res[i, j] = "%.4f±%.4f" % (arr_mean[i, j], arr_std[i, j])
    # res = pd.DataFrame(res, index=index, columns=columns)
    arr_mean = arr.mean(axis=0)
    res = pd.DataFrame(arr_mean, index=index, columns=columns)
    res["percent_bias"] = res["percent_bias"] * 100
    ind_pb = list(columns).index("percent_bias")
    res["se_precent_bias"] = arr[..., ind_pb].std(axis=0) / np.sqrt(
        arr.shape[0]
    )
    return res


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
    parser.add_argument("--summarize_save_fn", type=str, default=None)
    parser.add_argument("--summarize_target_pattern", type=str, default=None)

    parser.add_argument("--nrepeat", type=int, default=10)
    parser.add_argument("--ncores", type=int, default=1)
    parser.add_argument(
        "--solver",
        type=str,
        choices=["pymc", "blackjax", "numpyro", "nutpie", "vi"],
        default="pymc",
    )

    parser.add_argument(
        "--prevalence", type=float, nargs="+", default=[0.05, 0.25, 0.50]
    )
    parser.add_argument(
        "--OR", type=float, nargs="+", default=[1.25, 1.5, 1.75, 2, 2.25, 2.5]
    )
    parser.add_argument("--sigma2x", type=float, default=1.0)
    parser.add_argument("--sigma2e", type=float, nargs="+", default=[1.0])
    parser.add_argument("--a", type=float, nargs="+", default=[-3, 1, -1, 3])
    parser.add_argument(
        "--b", type=float, nargs="+", default=[0.5, 0.75, 1.25, 1.5]
    )
    parser.add_argument("--nSamples", type=int, nargs="+", default=[1000])
    parser.add_argument("--nKnowX", type=int, nargs="+", default=[100])
    parser.add_argument(
        "--prior_sigma_ws",
        type=str,
        choices=["gamma", "inv_gamma"],
        default="gamma",
    )
    parser.add_argument(
        "--prior_sigma_ab0",
        type=str,
        choices=["half_cauchy", "half_flat"],
        default="half_cauchy",
    )
    parser.add_argument("--use_hier_x_prior", action="store_true")
    parser.add_argument(
        "--direction", type=str, choices=["w->x", "x->w"], default="x->w"
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

    if args.summarize_target_pattern is not None:
        # 依靠pattern来找到要print的结果，而非通过指定的参数
        # 因为我们的参数是一直在递增的，所以通过指定的参数可能无法实现目的
        pattern = re.compile(re.escape(args.summarize_target_pattern))
        for fn in os.listdir(save_root):
            if fn.startswith("pipeline_") and fn.endswith(".h5"):
                search_res = pattern.search(fn)
                if search_res:
                    res_fn = osp.join(save_root, fn)
                    logger_main.info("The results of %s is:" % res_fn)
                    with h5py.File(res_fn, "r") as h5:
                        g_eva = h5["evaluate"]
                        summ_df = summarise_results(
                            g_eva["values"][:],
                            g_eva.attrs["index"],
                            g_eva.attrs["columns"],
                        )
                    if args.summarize_save_fn is not None:
                        summ_save_ffn = osp.join(
                            save_root, args.summarize_save_fn
                        )
                        # sheet name中不能有[]
                        stname = fn[9:-3].replace("[", "@").replace("]", "")
                        with pd.ExcelWriter(
                            summ_save_ffn,
                            mode="a" if osp.exists(summ_save_ffn) else "w",
                        ) as writer:
                            summ_df.to_excel(writer, sheet_name=stname)
                    print(summ_df)
        return

    for prev_i, or_i in product(args.prevalence, args.OR):
        trial_i = Trials(
            ndraws=1000,
            ntunes=1000,
            nrepeat=args.nrepeat,
            ncores=args.ncores,
            direction=args.direction,
            solver=args.solver,
            pytensor_cache=osp.expanduser("~/.pytensor/"),
            prevalence=prev_i,
            OR=or_i,
            sigma2_e=(
                args.sigma2e[0] if len(args.sigma2e) == 1 else args.sigma2e
            ),
            sigma2_x=args.sigma2x,
            a=args.a,
            b=args.b,
            n_sample_per_studies=(
                args.nSamples[0] if len(args.nSamples) == 1 else args.nSamples
            ),
            n_knowX_per_studies=(
                args.nKnowX[0] if len(args.nKnowX) == 1 else args.nKnowX
            ),
            prior_sigma_ws=args.prior_sigma_ws,
            prior_sigma_ab0=args.prior_sigma_ab0,
            use_hier_x_prior=args.use_hier_x_prior
        )
        if args.tasks == "simulate":
            save_fn = osp.join(
                save_root, "simulate_%s.h5" % trial_i._simul_name
            )
            if osp.exists(save_fn):
                if args.save_action == "raise":
                    raise FileExistsError(save_fn)
                elif args.save_action == "ignore":
                    logger_main.info("%s existed, skip." % save_fn)
                    continue
            arr, cols = trial_i.simulate()
            with h5py.File(save_fn, "w") as h5:
                g_sim = h5.create_group("simulate")
                g_sim.attrs["columns"] = cols
                g_sim.create_dataset("values", data=arr)
        elif args.tasks == "summarize":
            res_fn = osp.join(save_root, "pipeline_%s.h5" % trial_i._name)
            if not osp.exists(res_fn):
                logger_main.info("%s not exists, skip." % res_fn)
            else:
                logger_main.info("The results of %s is:" % res_fn)
                with h5py.File(res_fn, "r") as h5:
                    g_eva = h5["evaluate"]
                    summ_df = summarise_results(
                        g_eva["values"][:],
                        g_eva.attrs["index"],
                        g_eva.attrs["columns"],
                    )
                if args.summarize_save_fn is not None:
                    summ_save_ffn = osp.join(save_root, args.summarize_save_fn)
                    # sheet name中不能有[]
                    stname = trial_i._name.replace("[", "@").replace("]", "")
                    with pd.ExcelWriter(
                        summ_save_ffn,
                        mode="a" if osp.exists(summ_save_ffn) else "w",
                    ) as writer:
                        summ_df.to_excel(writer, sheet_name=stname)
                print(summ_df)
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


if __name__ == "__main__":
    main()
