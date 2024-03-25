import os
import os.path as osp
import re
# from typing import Dict, Tuple, Optional, List
from datetime import datetime

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15


# def parse_xlsx(fn: str) -> Dict[str, pd.DataFrame]:
#     xlsx = pd.read_excel(fn, sheet_name=None, header=[0, 1])
#     res = {}
#     for k, sheet in xlsx.items():
#         if not k.startswith("scenario"):
#             continue
#         sheet.iloc[:, :2] = sheet.iloc[:, :2].ffill(axis=0)
#         sheet = sheet.rename(columns={"Naive.1": "Bayes"})
#         index_cols = sheet.iloc[:, :2]
#         index_cols.columns = index_cols.columns.droplevel(1)
#         sheet = sheet.iloc[:, 2:]
#         sheet.index = pd.MultiIndex.from_frame(index_cols)
#         sheet = sheet.stack().reset_index()
#         colnames = sheet.columns.tolist()
#         colnames[1] = "OR"
#         colnames[2] = "Methods"
#         sheet.columns = [ci.strip() for ci in colnames]
#         sheet["OR"] = sheet["OR"].str.extract(r"log\((.*?)\)").astype(float)
#         res[k] = sheet
#     return res


# def plot_lines(
#     df: pd.DataFrame,
#     exclude_methods: Tuple[str] = (),
#     exclude_metrics: Tuple[str] = ("Bias", "Standard error"),
#     palette: Optional[List] = None,
# ) -> plt.figure:
#     df["Percent bias"] = df["Percent bias"].abs()
#     df.drop(columns=list(exclude_metrics), inplace=True)
#     df.columns = df.columns.str.strip()
#     df = df[~df["Methods"].isin(exclude_methods)]
#     df["Methods"] = df["Methods"].astype("category")

#     prevalances = df["Prevalence"].unique()
#     n_methods = df["Methods"].unique().shape[0]

#     fig = plt.figure(constrained_layout=True, figsize=(10, 10))
#     subfigs = fig.subfigures(3, 1, hspace=0.07)

#     for i, (metrici, metric_name) in enumerate(
#         zip(
#             ["Percent bias", "MSE", "Coverage rate"],
#             ["Percent Bias", "Mean Square Error", "Coverage Rate"],
#         )
#     ):
#         subfigs[i].suptitle(metric_name)
#         axs = subfigs[i].subplots(ncols=len(prevalances))
#         for j, ax in enumerate(axs):
#             dfi = df[df["Prevalence"] == prevalances[j]]
#             if metrici == "Coverage rate":
#                 dfi = dfi.query("Methods != 'Naive'")
#             sns.lineplot(
#                 data=dfi,
#                 x="OR",
#                 y=metrici,
#                 hue="Methods",
#                 ax=ax,
#                 palette=palette,
#             )
#             ax.set_xlabel("")
#             ax.set_ylabel("")
#             ax.set_yscale("log")
#             ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
#             ax.set_title("Prevalence = %.2f" % prevalances[j])

#     handles, labels = axs[0].get_legend_handles_labels()
#     fig.legend(
#         handles,
#         labels,
#         loc="outside lower center",
#         ncols=n_methods // 2,
#         frameon=False,
#         fancybox=False,
#     )

#     for subfig in subfigs:
#         for ax in subfig.axes:
#             ax.get_legend().remove()

#     return fig


def main():
    root = "./results/embp/"
    methods = ["naive", "xonly", "EMBP"]
    start = "2024-03-23_16-00-00"
    end = "2024-03-23_20-00-00"

    fmt = "%Y-%m-%d_%H-%M-%S"
    dt_start = datetime.strptime(start, fmt)
    dt_end = datetime.strptime(end, fmt)

    used_ffns = []
    for fn in os.listdir(root):
        sres = re.search(r"(2024-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}).nc", fn)
        if sres is not None:
            fdt = datetime.strptime(sres.group(1), fmt)
            if fdt >= dt_start and fdt <= dt_end:
                used_ffns.append(osp.join(root, fn))

    res = []
    for ffn in used_ffns:
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
            }
            resi.update(dt.attrs)
            res.append(resi)

    res = pd.DataFrame.from_records(res)
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
        fg.savefig(f"./results/embp/continue_sem_{metric}.png")


if __name__ == "__main__":
    main()
