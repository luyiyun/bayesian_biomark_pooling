import os
import os.path as osp
import sys
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

import sys
from bayesian_biomarker_pooling.simulate import (
    BinarySimulator,
    ContinuousSimulator,
    default_serializer,
)
from bayesian_biomarker_pooling import EMBP
from bayesian_biomarker_pooling.utils import Timer
import ipdb

# read data
file = "/mnt/e/SongJL/016_research/EMBP/case/GDCTCGA/ERBB2_simu.csv"
def embp_result(file):
    dat = pd.read_csv(file, index_col=0)
    X = dat["X"].values
    S = dat["S"].values
    W = dat["W"].values
    Y = dat["Y"].values
    Z = None
    # fit model
    embp_kwargs = {
                "ci": True,
                "ci_method": "bootstrap",
                "pbar": False,
                "max_iter": None,
                "seed": 0,
                "n_bootstrap": 200,
                "gem": False,
                "quasi_mc_K": 100,
                "delta2": None,
                "binary_solve": "lap",
                "device": "cpu",
                "importance_sampling_maxK": 5000,
            }
    estimator = EMBP("binary", **embp_kwargs)
    estimator.fit(X, S, W, Y, Z)
    # EMBP_result = estimator.params_.iloc[[14]]
    return estimator

# bbp实例10个抽样
result_simu = embp_result("/mnt/e/SongJL/016_research/EMBP/case/GDCTCGA/ERBB2_simu.csv")
result_simu
np.exp(result_simu)

# EMBP实例10个抽样
result_simu_10 = embp_result("/mnt/e/SongJL/016_research/EMBP/case/GDCTCGA/ERBB2_simu_10.csv")
result_simu_10
np.exp(result_simu_10)

# EMBP实例20个抽样
result_simu_20 = embp_result("/mnt/e/SongJL/016_research/EMBP/case/GDCTCGA/ERBB2_simu_20.csv")
result_simu_20
np.exp(result_simu_20)

# EMBP实例0.1比例抽样
result_simu_01 = embp_result("/mnt/e/SongJL/016_research/EMBP/case/GDCTCGA/ERBB2_simu_0.1.csv")
result_simu_01
np.exp(result_simu_01)

# EMBP实例0.2比例抽样
result_simu_02 = embp_result("/mnt/e/SongJL/016_research/EMBP/case/GDCTCGA/ERBB2_simu_0.2.csv")
result_simu_02
np.exp(result_simu_02)


# BRCA1-EMBP实例0.1比例抽样
result_BRCA1simu_01 = embp_result("/mnt/e/SongJL/016_research/EMBP/case/GDCTCGA/BRCA1_simu_0.1.csv")

result_BRCA1simu_01.res_bootstrap_
np.exp(result_BRCA1simu_01)

# BRCA1-EMBP实例0.2比例抽样
result_BRCA1simu_02 = embp_result("/mnt/e/SongJL/016_research/EMBP/case/GDCTCGA/BRCA1_simu_0.2.csv")
result_BRCA1simu_02
np.exp(result_BRCA1simu_02)

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
resi = method_xonly(X, Y, Z, "binary")
np.exp(resi)

dat1 = pd.read_csv("/mnt/e/SongJL/016_research/EMBP/bayesian_biomark_pooling/experiments/embp/scenario30051/data/binary_wo_z_100_0.3_1.5/data.csv")
dat2 = dat1[dat1["repeat"]==0]
X = dat2["X"]
S = dat2["S"]
W = dat2["W"]
Y = dat2["Y"]
Z = None

model = EMBP("binary")
model.fit(X, S, W, Y, Z)