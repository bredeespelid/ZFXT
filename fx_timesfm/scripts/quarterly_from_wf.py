from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fx_timesfm.evaluation import diebold_mariano


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate daily walk-forward (covariates) to quarter-end metrics and plot")
    ap.add_argument("--csv", required=True, help="Path to walk_forward_cov.csv with columns: timestamp, model, actual [optional: model_base]")
    ap.add_argument("--outdir", required=False, help="Output directory; defaults to CSV directory")
    args = ap.parse_args()

    csv_path = os.path.abspath(args.csv)
    out_dir = os.path.abspath(args.outdir or os.path.dirname(csv_path))
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if not set(["timestamp", "model", "actual"]).issubset(df.columns):
        raise SystemExit("CSV must contain columns: timestamp, model, actual")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").dropna(subset=["timestamp"]).reset_index(drop=True)

    # Quarter-end sampling: last available date in each quarter
    df["Q"] = df["timestamp"].dt.to_period("Q")
    qe_idx = df.groupby("Q")["timestamp"].idxmax()
    rq = df.loc[qe_idx].sort_values("timestamp").reset_index(drop=True)

    # RW baseline = previous day's actual level at QE
    rw_series = df.set_index("timestamp")["actual"].shift(1)
    rq["rw"] = rw_series.reindex(rq["timestamp"]).values

    mask = rq["actual"].notna() & rq["model"].notna() & rq["rw"].notna()
    rqv = rq[mask]
    y = rqv["actual"].to_numpy(dtype=float)
    yhat = rqv["model"].to_numpy(dtype=float)
    rw = rqv["rw"].to_numpy(dtype=float)

    obs = int(len(y))
    mae = float(np.mean(np.abs(yhat - y))) if obs else float("nan")
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2))) if obs else float("nan")
    dm_p = diebold_mariano(np.abs(yhat - y) - np.abs(rw - y), h=1) if obs else float("nan")

    # Save CSV and metrics
    rq_out = os.path.join(out_dir, "quarterly_cov.csv")
    rq[["timestamp", "model", "actual", "rw"]].to_csv(rq_out, index=False)
    with open(os.path.join(out_dir, "quarterly_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Observations: {obs}\n")
        f.write(f"MAE_model: {mae}\n")
        f.write(f"RMSE_model: {rmse}\n")
        f.write(f"DM_p (two-sided, abs loss): {dm_p}\n")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(rqv["timestamp"], rqv["actual"], label="Actual (EUR/NOK)", color="black")
    plt.plot(rqv["timestamp"], rqv["model"], label=" ", color="tab:blue", linestyle="--")
    plt.title("Quarterly Forecast (Covariates) – Actual vs Predicted (EUR/NOK)")
    plt.ylabel("Level (EUR/NOK)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_img = os.path.join(out_dir, "quarterly_actual_vs_pred_cov.jpeg")
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    try:
        plt.show()
    except Exception:
        pass

    print(f"Saved quarterly covariate metrics and plot under {out_dir}")


if __name__ == "__main__":
    main()
