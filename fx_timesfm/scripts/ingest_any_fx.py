from __future__ import annotations

import argparse
import os
from typing import List, Optional

import pandas as pd


def read_any(path: str, fmt: str) -> pd.DataFrame:
    if fmt == "auto":
        ext = os.path.splitext(path)[1].lower()
        fmt = "parquet" if ext in (".parquet", ".pq") else "csv"
    if fmt == "csv":
        return pd.read_csv(path)
    elif fmt == "parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError("Unsupported format")


def to_long(df: pd.DataFrame, timestamp: str, symbol: str, price: str) -> pd.DataFrame:
    out = df[[timestamp, symbol, price]].copy()
    out.columns = ["timestamp", "symbol", "price"]
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out = out.dropna(subset=["timestamp", "price"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return out


def resample_per_symbol(df: pd.DataFrame, freq: str, fill: str) -> pd.DataFrame:
    outs = []
    for sym, g in df.groupby("symbol"):
        g2 = g.set_index("timestamp").sort_index()[["price"]].asfreq(freq)
        if fill == "ffill":
            g2 = g2.ffill()
        elif fill == "mean":
            g2 = g2.fillna(g2["price"].mean())
        elif fill == "last":
            g2 = g2.fillna(method="bfill").fillna(method="ffill")
        g2["symbol"] = sym
        outs.append(g2.reset_index())
    return pd.concat(outs, ignore_index=True)


def split_no_leak(df: pd.DataFrame, val_ratio: float, test_ratio: float):
    outs = {"train": [], "val": [], "test": []}
    for sym, g in df.groupby("symbol"):
        n = len(g)
        i1 = int(n * (1 - (val_ratio + test_ratio)))
        i2 = int(n * (1 - test_ratio))
        outs["train"].append(g.iloc[:i1])
        outs["val"].append(g.iloc[i1:i2])
        outs["test"].append(g.iloc[i2:])
    return {k: pd.concat(v).reset_index(drop=True) for k, v in outs.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--format", choices=["csv", "parquet", "auto"], default="auto")
    p.add_argument("--timestamp", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--price", required=True)
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--freq", choices=["D", "H", "M"], default="D")
    p.add_argument("--outdir", default="data_any")
    p.add_argument("--symbols", nargs="*")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--fill", choices=["ffill", "mean", "last"], default="ffill")
    args = p.parse_args()

    df = read_any(args.input, args.format)
    df = to_long(df, args.timestamp, args.symbol, args.price)
    if args.symbols:
        df = df[df["symbol"].isin(set(args.symbols))]
    # filter >= start
    df = df[df["timestamp"] >= pd.to_datetime(args.start, utc=True)]
    # resample per symbol
    df = resample_per_symbol(df, args.freq, args.fill)

    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, "long.csv"), index=False)
    splits = split_no_leak(df, args.val_ratio, args.test_ratio)
    for k, d in splits.items():
        d.to_csv(os.path.join(args.outdir, f"{k}.csv"), index=False)
        print(f"wrote {k}.csv rows={len(d)}")


if __name__ == "__main__":
    main()
