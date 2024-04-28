import os
import os.path as osp
import re
import json
from datetime import datetime
from argparse import ArgumentParser

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
    ) -> None:
        self._h, self._w = height, width
        self._palette = palette
        # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        self._d = d

    def plot(
        self, df: pd.DataFrame, x: str, y: str, hue: str, col: str, row: str
    ) -> tuple[Figure, Axes]:
        col_uni = np.sort(df[col].unique())
        row_uni = np.sort(df[row].unique())

        self._fig = plt.figure(
            constrained_layout=True,
            figsize=(self._w * len(col_uni), self._h * len(row_uni)),
        )
        subfigs = self._fig.subfigures(nrows=len(row_uni), ncols=len(col_uni))
        self._handles, self._labels = [], []

        for i, ri in enumerate(row_uni):
            for j, cj in enumerate(col_uni):
                dfi = df[(df[col] == cj) & (df[row] == ri)]
                figi = subfigs[i, j]
                self._subfig_plot_line(figi, dfi, x, y, hue)

                ri_str = f"{ri: .3f}" if isinstance(ri, float) else ri
                cj_str = f"{cj: .3f}" if isinstance(cj, float) else cj
                figi.suptitle(f"{row}={ri_str}, {col}={cj_str}")

        self._fig.supylabel(y)
        self._fig.supxlabel(x)

        self._fig.legend(
            handles=self._handles,
            labels=self._labels,
            ncols=len(self._labels),
            loc="outside lower center",
            # loc="lower right",
            # markerscale=markerscale,
            # frameon=False,
            # fancybox=False,
            # bbox_to_anchor=(0.4, -0.2, 0.2, 0.2),
            # columnspacing=0.2,
            # handletextpad=0.1,
        )

        return self._fig

    def _subfig_plot_line(
        self, fig: Figure, dfi: pd.DataFrame, x: str, y: str, hue: str
    ):
        # 首先根据hue进行划分，得到每个hue所代表的内容
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
            (handle,) = ax.plot(x, y, "-", c=self._hue_map[hi], label=hi)
            if hi not in self._labels:
                self._handles.append(handle)
                self._labels.append(hi)
        ax.set_ylim(ymin, ymax)


def plot_lines(
    name: str | None,
    start: str | None = None,
    end: str | None = None,
    root: str = "./results/embp/",
):
    methods = ["naive", "xonly", "EMBP"]

    fmt = "%Y-%m-%d_%H-%M-%S"
    dt_start = (
        datetime.fromisoformat("2024-01-01")
        if start is None
        else datetime.strptime(start, fmt)
    )
    dt_end = datetime.now() if end is None else datetime.strptime(end, fmt)

    res = []
    for fn in os.listdir(root):
        sres = re.search(r"(2024-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}).nc", fn)
        if sres is None:
            continue
        fdt = datetime.strptime(sres.group(1), fmt)
        if fdt < dt_start or fdt > dt_end:
            continue
        if name is not None and name not in fn:
            continue

        # 读取json文件
        with open(osp.join(root, fn[:-3] + ".json"), "r") as f:
            attrs = json.load(f)

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
                "n_sample_per_studies": attrs["nsample"],
                "x_ratio": attrs["ratiox"],
            }
            res.append(resi)

    res = pd.DataFrame.from_records(res)
    res["x_ratio"] = res["x_ratio"].astype("category")
    # bias计算为绝对值
    res["bias"] = res["bias"].abs()
    print(res.to_string())

    plottor = BrokenLinePlottor(3, 3)
    for metric in ["bias", "mse", "cov_rate"]:
        fg = plottor.plot(
            res,
            x="n_sample_per_studies",
            y=metric,
            hue="method",
            col="x_ratio",
            row="beta_x",
        )
        fg.savefig(osp.join(root, f"{name}_{metric}.png"))


def main():
    parser = ArgumentParser()
    parser.add_argument("--root", default="./results/embp/")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    plot_lines(name=args.name, start=args.start, end=args.end, root=args.root)


if __name__ == "__main__":
    main()
