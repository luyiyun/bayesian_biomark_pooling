import os
import os.path as osp
import re
from datetime import datetime

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15


def plot_lines(
    name: str,
    start: str,
    end: str,
    root: str = "./results/embp/",
    calc_x_ratio: bool = False,
):
    root = "./results/embp/"
    methods = ["naive", "xonly", "EMBP"]

    fmt = "%Y-%m-%d_%H-%M-%S"
    dt_start = datetime.strptime(start, fmt)
    dt_end = datetime.strptime(end, fmt)

    res = []
    for fn in os.listdir(root):
        sres = re.search(r"(2024-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}).nc", fn)
        if sres is None:
            continue

        fdt = datetime.strptime(sres.group(1), fmt)
        if fdt < dt_start or fdt > dt_end:
            continue

        ffn = osp.join(root, fn)
        dt = xr.load_dataset(ffn)
        da_true = dt["true"].sel(params="beta_x").item()

        for key in methods:
            diff = dt[key].sel(params="beta_x", statistic="estimate") - da_true
            # bias
            bias_mean = diff.mean().item()
            bias_std = diff.std().item()
            # mse
            mse = (diff**2).mean().item()
            # cov rate
            covr = (
                (
                    (dt[key].sel(params="beta_x", statistic="CI_1") <= da_true)
                    & (
                        dt[key].sel(params="beta_x", statistic="CI_2")
                        >= da_true
                    )
                )
                .mean()
                .item()
            )

            resi = {
                "bias": bias_mean,
                "bias_std": bias_std,
                "mse": mse,
                "cov_rate": covr,
                "beta_x": da_true,
                "method": key,
                "datetime": fdt,
            }
            resi.update(dt.attrs)
            res.append(resi)

    res = pd.DataFrame.from_records(res)
    if calc_x_ratio:
        # 重新计算x_ratio
        res["x_ratio"] = (
            res["n_knowX_per_studies"] / res["n_sample_per_studies"]
        )
        res.loc[
            (res["x_ratio"] >= 0.14) & (res["x_ratio"] <= 0.15), "x_ratio"
        ] = 0.15
        res["x_ratio"] = res["x_ratio"].astype("category")
    # bias计算为绝对值
    res["bias"] = res["bias"].abs()
    print(res.to_string())

    for metric in ["bias", "mse", "cov_rate"]:
        fg = sns.relplot(
            data=res,
            kind="line",
            x="n_sample_per_studies",
            y=metric,
            hue="method",
            col="x_ratio",
            row="beta_x",
            facet_kws={"sharey": False, "sharex": False},
        )
        fg.set(yscale="log")
        fg.savefig(f"./results/embp/{name}_{metric}.png")


def main():
    plot_lines(
        name="continue_bootstrap",
        start="2024-03-23_17-00-00",
        end="2024-03-23_20-00-00",
        calc_x_ratio=True,
    )

    plot_lines(
        name="continue_bootstrap",
        start="2024-03-24_17-00-00",
        end="2024-03-25_10-00-00",
        calc_x_ratio=True,
    )


if __name__ == "__main__":
    main()
