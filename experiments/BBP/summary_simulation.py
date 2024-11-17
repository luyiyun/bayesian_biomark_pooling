import os.path as osp
import re
from glob import glob
from argparse import ArgumentParser

import pandas as pd
import xarray as xr


def main():
    parser = ArgumentParser()
    parser.add_argument("--res_root", type=str, default="./results/")
    parser.add_argument(
        "--res_files", type=str, default="weak_info_wo_multi_imp_binary-*.nc"
    )
    parser.add_argument(
        "--save_fn", type=str, default="./results/summary_res.xlsx"
    )
    parser.add_argument("--data_root", type=str, default="./data/")
    # parser.add_argument(
    #     "--index_configs",
    #     type=str,
    #     nargs="+",
    #     default=["beta_x"],
    # )
    # parser.add_argument(
    #     "--column_configs",
    #     type=str,
    #     nargs="+",
    #     default=["beta_0"],
    # )
    parser.add_argument(
        "--index_name_configs",
        type=str,
        nargs="+",
        default=["OR"],
    )
    parser.add_argument(
        "--column_name_configs",
        type=str,
        nargs="+",
        default=["Prev"],
    )
    args = parser.parse_args()

    files = glob(osp.join(args.res_root, args.res_files))

    df = []
    for fi in files:
        dati = xr.load_dataset(fi)
        dfi = dati["score"].to_dataframe().T

        # res = re.search(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", fi)
        # str_date = res.group(0)
        # config_file = glob(osp.join(args.data_root, f"*{str_date}*.json"))
        # assert len(config_file) == 1
        # with open(config_file[0], "r") as jf:
        #     config = json.load(jf)

        # for confi in (args.index_configs + args.column_configs):
        #     val = config[confi]
        #     if isinstance(val, list):
        #         if len(set(val)) == 1:
        #             val = val[0]
        #         else:
        #             val = "-".join(str(val))
        #     dfi[confi] =val

        for confi in args.index_name_configs + args.column_name_configs:
            res = re.search(rf"[-_]{confi}(.*?)[-_]", fi)
            if res is not None:
                val = float(res.group(1).replace(",", "."))
            else:
                val = ""
            dfi[confi] = val

        df.append(dfi)

    df = pd.concat(df)
    df = df.pivot(
        index=args.index_name_configs, columns=args.column_name_configs
    )

    with pd.ExcelWriter(
        args.save_fn,
        mode="a" if osp.exists(args.save_fn) else "w",
        if_sheet_exists="replace",
    ) as writer:
        df.to_excel(writer, sheet_name=args.res_files.replace("*", "X"))


if __name__ == "__main__":
    main()
