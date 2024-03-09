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
    dat = pd.read_csv(osp.join(save_root, "ERBB2_simu.csv"), index_col=0)

    # fit model
    model = BBP(
        ntunes=5000,
        nsample=5000,
        prior_x=(0, 10, 2, 1),
        prior_a=(0, 10, 2, 1),
        prior_b=(0, 10, 2, 1),
        prior_beta0=(0, 10, 2, 1),
        prior_sigma=(2, 1),
        nchains=1
    )

    model.fit(dat)
    baye_res = model.summary()
    print(baye_res)
    baye_res.to_csv(osp.join(save_root, "ERBB2_res.csv"))

    model.plot("./results/ERBB2_plot")


if __name__ == "__main__":
    main()
