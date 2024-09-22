import logging
import os
import os.path as osp

import pandas as pd
from bayesian_biomarker_pooling import BBP

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
    save_root = "./results/"
    os.makedirs(save_root, exist_ok=True)

    # read data
    dat = pd.read_csv("./data/ERBB2_simu.csv", index_col=0)
    dat["S"] = dat["S"].astype(int)

    # fit model
    model = BBP()

    fit_res = model.fit(dat)
    summary_res = fit_res.summary()
    print(summary_res)
    summary_res.to_csv(osp.join(save_root, "ERBB2_res.csv"))


if __name__ == "__main__":
    main()
