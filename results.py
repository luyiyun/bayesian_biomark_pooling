import os
import os.path as osp
import re
import json
from datetime import datetime

# from argparse import ArgumentParser
from typing import Sequence

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10


class BrokenLinePlottor:

    def __init__(
        self,
        height: float,
        width: float,
        palette: str | None = None,
        d: float = 0.015,
        height_pad: float = 0.0,
        width_pad: float = 0.0,
    ) -> None:
        self._h, self._w = height, width
        self._hp, self._wp = height_pad, width_pad
        self._palette = palette
        # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        self._d = d

    def plot(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        hue: str,
        col: str,
        row: str,
        hue_order: Sequence[str] | None = None,
        ylabel: str | None = None,
        xlabel: str | None = None,
        col_label: str | None = None,
        row_label: str | None = None,
        legend: bool = False,
    ) -> tuple[Figure, Axes]:
        self._hue_order = hue_order

        col_label = col_label or (f"{col} = " + "{col:.2f}")
        row_label = row_label or (f"{row} = " + "{row:.2f}")

        col_uni = np.sort(df[col].unique())
        row_uni = np.sort(df[row].unique())

        self._fig = plt.figure(
            # constrained_layout=True,
            layout="compressed",
            figsize=(
                self._w * len(col_uni) + self._wp,
                self._h * len(row_uni) + self._hp,
            ),
        )
        subfigs = self._fig.subfigures(
            nrows=len(row_uni),
            ncols=len(col_uni),
        )
        self._handles, self._labels = [], []

        for i, ri in enumerate(row_uni):
            for j, cj in enumerate(col_uni):
                dfi = df[(df[col] == cj) & (df[row] == ri)]
                figi = subfigs[i, j]
                self._subfig_plot_line(figi, dfi, x, y, hue)

                if i == 0:
                    figi.supxlabel(col_label.format(col=cj), y=0.98)
                if j == (len(col_uni) - 1):
                    figi.supylabel(
                        row_label.format(row=ri), x=0.98, rotation=270
                    )

        self._fig.supylabel(ylabel or y)
        self._fig.supxlabel(xlabel or x)
        # 会缩小figure中的内容，从而让figi的supxlabel和supylabel显示出来
        self._fig.get_layout_engine().set(rect=((0, 0, 0.98, 0.98)))

        if legend:
            self._fig.legend(
                handles=self._handles,
                labels=self._labels,
                ncols=1,
                loc="outside right",
                # loc="lower right",
                # markerscale=markerscale,
                frameon=False,
                fancybox=False,
                # bbox_to_anchor=(0.4, -0.2, 0.2, 0.2),
                # columnspacing=0.2,
                # handletextpad=0.1,
            )

        return self._fig

    def _subfig_plot_line(
        self, fig: Figure, dfi: pd.DataFrame, x: str, y: str, hue: str
    ):
        # 首先根据hue进行划分，得到每个hue所代表的内容
        if self._hue_order is not None:
            hue_uni = self._hue_order
        else:
            hue_uni = dfi[hue].unique()
        self._hue_map = {
            k: v for k, v in zip(hue_uni, sns.color_palette(self._palette))
        }
        y_intervals = []
        for hi in hue_uni:
            dfi_hi = dfi[dfi[hue] == hi]
            y_hi = dfi_hi[y].values
            y_hi_max, y_hi_min = y_hi.max(), y_hi.min()
            y_intervals.append(
                (
                    y_hi_min,
                    y_hi_max,
                    y_hi_max - y_hi_min,
                    hi,
                    dfi_hi,
                )
            )
        y_intervals = sorted(y_intervals, key=lambda x: x[0])

        # 我们确定哪些hue要放在一个子图，并且确定每个子图所占的比例
        sep = []
        for i, (y_inte_1, y_inte_2) in enumerate(
            zip(y_intervals[:-1], y_intervals[1:])
        ):
            if (y_inte_2[0] - y_inte_1[1]) > max(
                y_inte_1[2], y_inte_2[2]
            ) * 1.5:
                sep.append(i + 1)
        sep = [0] + sep + [len(y_intervals)]

        dfi_s, heights, lows, tops = [], [], [], []
        for start, end in zip(sep[:-1], sep[1:]):
            ymins, ymaxs, dfi_s_i = [], [], []
            for yi in y_intervals[start:end]:
                ymins.append(yi[0])
                ymaxs.append(yi[1])
                dfi_s_i.append((yi[3], yi[4]))
            lows.append(min(ymins))
            tops.append(max(ymaxs))
            heights.append(max(ymaxs) - min(ymins))
            dfi_s.append(dfi_s_i)
        # heights将每个元素拿到，最小也必须是最大的1/10
        height_max = max(heights)
        pad = height_max / 10
        heights = [max(h, pad) for h in heights]
        lows = [t - pad / 2 for t in lows]
        tops = [t + pad / 2 for t in tops]

        # 反向
        dfi_s.reverse()
        tops.reverse()
        lows.reverse()
        heights.reverse()
        # 绘制每个子图
        axs = fig.subplots(
            nrows=len(dfi_s), height_ratios=heights, squeeze=False
        )
        for i, axi in enumerate(axs.flatten()):
            # 取消掉上边界和下边界，并在y轴加入短斜线
            kwargs = dict(transform=axi.transAxes, color="k", clip_on=False)
            if i > 0:
                axi.spines["top"].set_visible(False)
                axi.plot(
                    (-self._d, +self._d), (1 - self._d, 1 + self._d), **kwargs
                )  # bottom-left diagonal
                axi.plot(
                    (1 - self._d, 1 + self._d),
                    (1 - self._d, 1 + self._d),
                    **kwargs,
                )  # bottom-right diagonal
            if i < (len(axs) - 1):
                axi.spines["bottom"].set_visible(False)
                axi.tick_params(bottom=False, labelbottom=False)
                axi.plot(
                    (-self._d, +self._d), (-self._d, +self._d), **kwargs
                )  # top-left diagonal
                axi.plot(
                    (1 - self._d, 1 + self._d), (-self._d, +self._d), **kwargs
                )  # top-right diagonal

            # 绘制图形
            self._ax_plot_minor(axi, dfi_s[i], x, y, lows[i], tops[i])

    def _ax_plot_minor(
        self,
        ax: Axes,
        dfi_s_i: list[tuple[str, pd.DataFrame]],
        x: str,
        y: str,
        ymin: float,
        ymax: float,
    ):
        triple = [(hi, dfii[x].values, dfii[y].values) for hi, dfii in dfi_s_i]

        for hi, x, y in triple:
            ind = np.argsort(x)
            x, y = x[ind], y[ind]
            (handle,) = ax.plot(x, y, ".-", c=self._hue_map[hi], label=hi)
            if hi not in self._labels:
                self._handles.append(handle)
                self._labels.append(hi)
        ax.set_ylim(ymin, ymax)


def load_results(
    root: str = "./results/embp/",
    methods: Sequence[str] | dict[str, str] = ("naive", "xonly", "EMBP"),
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    fmt = "%Y-%m-%d_%H-%M-%S"
    dt_start = (
        datetime.fromisoformat("2024-01-01")
        if start is None
        else datetime.strptime(start, fmt)
    )
    dt_end = datetime.now() if end is None else datetime.strptime(end, fmt)

    if isinstance(methods, dict):
        methods_iter = methods.items()
    else:
        methods_iter = list(zip(methods, methods))

    res = []
    for fn in os.listdir(root):
        sres = re.search(r"(2024-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}).nc", fn)
        if sres is None:
            continue
        fdt = datetime.strptime(sres.group(1), fmt)
        if fdt < dt_start or fdt > dt_end:
            continue

        # 读取json文件
        with open(osp.join(root, fn[:-3] + ".json"), "r") as f:
            attrs = json.load(f)

        ffn = osp.join(root, fn)
        dt = xr.load_dataset(ffn)
        da_true = dt["true"].sel(params="beta_x").item()

        for key, value in methods_iter:
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
                "method": value,
                "datetime": fdt,
                "n_sample_per_studies": attrs["nsample"],
                "x_ratio": attrs["ratiox"],
            }
            res.append(resi)

    res = pd.DataFrame.from_records(res)
    return res


def continue_wo_z():
    name = "continue_wo_z"
    res = pd.concat(
        [
            load_results("./results/embp/continue_wo_z/"),
            load_results(
                "./results/continue_wo_z_EMBP_sem/",
                methods={"EMBP": "EMBP-sem"},
            ),
        ]
    )

    res_summ = res.copy()
    res_summ["bias"] = [
        f"{a:.4f}±{b:.4f}"
        for a, b in zip(res_summ["bias"], res_summ["bias_std"])
    ]
    res_summ_bias_mse = (
        res_summ.drop(columns=["datetime", "bias_std", "cov_rate"])
        .query("method != 'EMBP-sem'")
        .set_index(
            ["beta_x", "x_ratio", "n_sample_per_studies", "method"],
            drop=True,
        )
        .unstack(level="n_sample_per_studies")
    )
    res_summ_cov_rate = (
        res_summ.drop(columns=["datetime", "bias_std", "bias", "mse"])
        .replace({"method": {"EMBP": "EMBP-is"}})
        .set_index(
            ["beta_x", "x_ratio", "n_sample_per_studies", "method"],
            drop=True,
        )
        .unstack(level="n_sample_per_studies")
    )
    with pd.ExcelWriter(f"./results/{name}.xlsx") as writer:
        res_summ_bias_mse.to_excel(writer, sheet_name="summary_bias_mse")
        res_summ_cov_rate.to_excel(writer, sheet_name="summary_cov_rate")
        res.to_excel(writer, sheet_name="raw")
    print(res_summ_bias_mse.to_string())
    print(res_summ_cov_rate.to_string())

    # figures
    # 绘图前记得做如下操作
    res["x_ratio"] = res["x_ratio"].astype("category")
    res["bias"] = res["bias"].abs()
    plottor = BrokenLinePlottor(3, 3, width_pad=1)
    for metric in ["bias", "mse", "cov_rate"]:
        fg = plottor.plot(
            (
                res.replace({"method": {"EMBP": "EMBP-is"}})
                if metric == "cov_rate"
                else res.query("method != 'EMBP-sem'")
            ),
            x="n_sample_per_studies",
            y=metric,
            hue="method",
            col="x_ratio",
            row="beta_x",
            xlabel="The number of samples for each study",
            ylabel={
                "bias": "Absolute Mean of Bias",
                "mse": "Mean Square Error",
                "cov_rate": "Covarage Rate",
            }[metric],
            col_label=r"The ratio of $X^o$ is {col:.2f}",
            row_label=r"The $\beta_x$ is {row:.1f}",
            hue_order=(
                ["naive", "xonly", "EMBP-sem", "EMBP-is"]
                if metric == "cov_rate"
                else ["naive", "xonly", "EMBP"]
            ),
            legend=True,
        )
        fg.savefig(f"./results/{name}_{metric}.png", pad_inches=0.5)


def main():
    # parser = ArgumentParser()
    # parser.add_argument("--root", default=["./results/embp/"], nargs="+")
    # parser.add_argument("--start", default=None)
    # parser.add_argument("--end", default=None)
    # parser.add_argument("--save_fn", default=None)
    # args = parser.parse_args()

    continue_wo_z()

    # res_binary_wo_z = pd.concat(
    #     [
    #         load_results(
    #             "./results/binary_wo_z_qK100_fix/",
    #             methods={
    #                 "naive": "naive",
    #                 "xonly": "xonly",
    #                 "EMBP": "EMBP-lap",
    #             },
    #         ),
    #         load_results(
    #             "./results/binary_IS_wo_z_nr200_1/",
    #             methods={"EMBP": "EMBP-is"},
    #         ),
    #         load_results(
    #             "./results/binary_IS_wo_z_nr200_2/",
    #             methods={"EMBP": "EMBP-is"},
    #         ),
    #     ]
    # )

    # for name, res in zip(
    #     ["continue_wo_z", "binary_wo_z"], [res_continue_wo_z, res_binary_wo_z]
    # ):
    #     res_summ = res.copy()
    #     res_summ["bias"] = [
    #         f"{a:.4f}±{b:.4f}"
    #         for a, b in zip(res_summ["bias"], res_summ["bias_std"])
    #     ]
    #     res_summ = (
    #         res_summ.drop(columns=["datetime", "bias_std"])
    #         .set_index(
    #             ["beta_x", "x_ratio", "n_sample_per_studies", "method"],
    #             drop=True,
    #         )
    #         .unstack(level="n_sample_per_studies")
    #     )
    #     with pd.ExcelWriter(f"./results/{name}.xlsx") as writer:
    #         res_summ.to_excel(writer, sheet_name="summary")
    #         res.to_excel(writer, sheet_name="raw")
    #     print(res_summ.to_string())

    #     # figures
    #     # 绘图前记得做如下操作
    #     res["x_ratio"] = res["x_ratio"].astype("category")
    #     res["bias"] = res["bias"].abs()
    #     plottor = BrokenLinePlottor(3, 3, width_pad=1)
    #     for metric in ["bias", "mse", "cov_rate"]:
    #         fg = plottor.plot(
    #             res,
    #             x="n_sample_per_studies",
    #             y=metric,
    #             hue="method",
    #             col="x_ratio",
    #             row="beta_x",
    #             xlabel="The number of samples for each study",
    #             ylabel={
    #                 "bias": "The mean of Bias",
    #                 "mse": "MSE",
    #                 "cov_rate": "The Covarage Rate",
    #             }[metric],
    #             col_label=r"The ratio of $X^o$ is {col:.2f}",
    #             row_label=r"The $\beta_x$ is {row:.1f}",
    #             hue_order=(
    #                 ["naive", "xonly", "EMBP"]
    #                 if name.startswith("continue")
    #                 else ["naive", "xonly", "EMBP-is", "EMBP-lap"]
    #             ),
    #             legend=True,
    #         )
    #         fg.savefig(f"./results/{name}_{metric}.png", pad_inches=0.5)


if __name__ == "__main__":
    main()
