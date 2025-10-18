from __future__ import annotations

import argparse
import os
import pandas as pd

from fx_timesfm.evaluation import evaluate_monthly_quarterly


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=False)  # not used for RW baseline; present for parity
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--freqs", default="M,Q")
    p.add_argument("--data_csv", default="data_any/val.csv")
    args = p.parse_args()

    df = pd.read_csv(args.data_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    res = evaluate_monthly_quarterly(df)
    print(res)


if __name__ == "__main__":
    main()
