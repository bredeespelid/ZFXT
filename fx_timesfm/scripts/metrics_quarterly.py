from __future__ import annotations

import argparse
import os
import math
import numpy as np
import pandas as pd

from fx_timesfm.evaluation import diebold_mariano


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute Observations, MAE, RMSE, and DM p-value from quarterly_eval.csv")
    ap.add_argument("--csv", type=str, required=False, help="Path to quarterly_eval.csv")
    ap.add_argument("--outdir", type=str, required=False, help="Folder containing quarterly_eval.csv; also where metrics will be saved")
    ap.add_argument("--loss", type=str, choices=["abs", "sq"], default="abs", help="Loss for DM test (abs or squared error)")
    ap.add_argument("--nw_lag", type=int, default=None, help="Newey–West lag/bandwidth; default h-1")
    args = ap.parse_args()

    if not args.csv and not args.outdir:
        raise SystemExit("Provide --csv or --outdir")

    if args.csv:
        csv_path = args.csv
        out_dir = args.outdir or os.path.dirname(os.path.abspath(csv_path))
    else:
        out_dir = args.outdir
        csv_path = os.path.join(out_dir, "quarterly_eval.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in ("model", "rw", "actual"):
        if col not in df.columns:
            raise ValueError("CSV must contain columns: model, rw, actual")

    # Keep rows with finite actuals and predictions
    m = np.isfinite(df["actual"]) & np.isfinite(df["model"]) & np.isfinite(df["rw"])  # type: ignore[arg-type]
    d = df[m].copy()
    if d.empty:
        raise SystemExit("No valid rows to evaluate (check for NaNs)")

    y = d["actual"].astype(float).to_numpy()
    yhat = d["model"].astype(float).to_numpy()
    rw = d["rw"].astype(float).to_numpy()

    # Metrics
    n_obs = int(len(y))
    mae = float(np.mean(np.abs(yhat - y)))
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))

    # DM test: model vs RW
    if args.loss == "abs":
        losses_model = np.abs(yhat - y)
        losses_rw = np.abs(rw - y)
    else:
        losses_model = (yhat - y) ** 2
        losses_rw = (rw - y) ** 2
    diff = losses_model - losses_rw
    dm_p = diebold_mariano(diff.astype(float), h=1, nw_lag=args.nw_lag)

    # Save
    out_path = os.path.join(out_dir, "metrics_extra.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Observations: {n_obs}\n")
        f.write(f"MAE_model: {mae}\n")
        f.write(f"RMSE_model: {rmse}\n")
        f.write(f"DM_p (two-sided, loss={args.loss}): {dm_p}\n")

    print(f"Observations: {n_obs}")
    print(f"MAE_model: {mae:.6f}")
    print(f"RMSE_model: {rmse:.6f}")
    print(f"DM_p (two-sided, loss={args.loss}): {dm_p:.12e}")
    print(f"Saved extra metrics to {out_path}")


if __name__ == "__main__":
    main()
