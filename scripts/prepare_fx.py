from __future__ import annotations

import argparse
import os

import pandas as pd

from fx_timesfm.data import load_price_frames, resample_uniform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--outdir", type=str, default="data")
    parser.add_argument("--freq", type=str, default="1H")
    parser.add_argument("--timestamp_col", type=str, default="timestamp")
    parser.add_argument("--symbol_col", type=str, default="symbol")
    parser.add_argument("--price_col", type=str, default="price")
    args = parser.parse_args()

    df = load_price_frames(args.inputs, args.timestamp_col, args.symbol_col, args.price_col)
    df = resample_uniform(df, args.freq, args.timestamp_col, args.symbol_col, args.price_col)
    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, "all.csv"), index=False)
    # naive temporal split 80/10/10 per symbol
    outs = {}
    for sym, g in df.groupby(args.symbol_col):
        n = len(g)
        i1 = int(n * 0.8)
        i2 = int(n * 0.9)
        outs.setdefault("train", []).append(g.iloc[:i1])
        outs.setdefault("val", []).append(g.iloc[i1:i2])
        outs.setdefault("test", []).append(g.iloc[i2:])
    for k, lst in outs.items():
        pd.concat(lst).to_csv(os.path.join(args.outdir, f"{k}.csv"), index=False)
        print(f"wrote {k}.csv")


if __name__ == "__main__":
    main()
