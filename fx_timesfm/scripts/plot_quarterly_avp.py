from __future__ import annotations

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot Actual vs Predicted from quarterly_eval.csv (no RW), save as JPEG."
    )
    ap.add_argument("--csv", type=str, required=False, help="Path to quarterly_eval.csv")
    ap.add_argument("--outdir", type=str, required=False, help="Directory containing quarterly_eval.csv and where to save plot")
    ap.add_argument("--out", type=str, default=None, help="Output image path (JPEG/PNG). Default: <outdir>/quarterly_actual_vs_pred.jpeg")
    ap.add_argument("--pair", type=str, default="EUR/NOK", help="Pair label for titles, e.g., 'EUR/NOK'")
    ap.add_argument("--title", type=str, default=None, help="Optional custom title")
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
    if "quarter_end" not in df.columns or "model" not in df.columns or "actual" not in df.columns:
        raise ValueError("CSV must contain columns: quarter_end, model, actual")

    # Build time index
    ts = pd.to_datetime(df["quarter_end"], utc=True, errors="coerce")
    y_pred = df["model"].astype(float)
    y_act = df["actual"].astype(float)

    # Drop NaNs in actuals (if any)
    m = y_act.notna()
    ts, y_pred, y_act = ts[m], y_pred[m], y_act[m]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(ts, y_act, label=f"Actual ({args.pair})", color="black")
    plt.plot(ts, y_pred, label=" ", color="tab:blue", linestyle="--")

    title = args.title or f"Quarterly Forecast: Actual vs Predicted ({args.pair})"
    plt.title(title)
    plt.ylabel(f"Level ({args.pair})")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = args.out or os.path.join(out_dir, "quarterly_actual_vs_pred.jpeg")
    # Save before show to ensure file is written in headless environments
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    try:
        plt.show()
    except Exception:
        # In headless environments, show() may fail; ignore
        pass
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
