from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from fx_timesfm.checkpoints import load_ckpt
from fx_timesfm.data_sources.norgesbank import fetch_fx_pair
from fx_timesfm.model import DecoderOnlyTSModel
from fx_timesfm.plot_utils import plot_forecast
from fx_timesfm.resample import to_freq
from fx_timesfm.evaluation import diebold_mariano


def build_model(ckpt_path: str) -> tuple[DecoderOnlyTSModel, dict, torch.device]:
    payload, card = load_ckpt(ckpt_path)
    cfg = payload.get("cfg", {})
    mcfg = cfg.get("model", {})
    dcfg = cfg.get("data", {})
    model = DecoderOnlyTSModel(
        input_patch_len=dcfg.get("input_patch_len", 32),
        output_patch_len=dcfg.get("output_patch_len", 128),
        d_model=mcfg.get("d_model", 512),
        n_layers=mcfg.get("n_layers", 8),
        n_heads=mcfg.get("n_heads", 8),
        dropout=mcfg.get("dropout", 0.2),
        quantiles=mcfg.get("quantiles", False),
        quantile_levels=mcfg.get("quantile_levels", [0.1, 0.5, 0.9]),
    )
    model.load_state_dict(payload["model_state"])  # type: ignore[arg-type]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, dcfg, device


def scalar_stats(patches: torch.Tensor) -> tuple[float, float]:
    mean = patches[:, :1, :].mean(dim=(1, 2), keepdim=True)
    std = patches[:, :1, :].std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return float(mean.view(-1)[0].item()), float(std.view(-1)[0].item())


def quarter_boundaries(start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    # Returns list of (Q_start, Q_end) inclusive
    # Use pandas period_range for quarter starts
    qs = pd.period_range(start=start, end=end, freq="Q")
    bounds: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for q in qs:
        q_end = q.end_time.tz_convert("UTC") if q.end_time.tzinfo else q.end_time.tz_localize("UTC")
        q_start = (q - 1).end_time + pd.Timedelta(seconds=1)
        q_start = q_start.tz_convert("UTC") if q_start.tzinfo else q_start.tz_localize("UTC")
        bounds.append((q_start.normalize(), q_end.normalize()))
    return bounds


def find_origin(df_daily: pd.DataFrame, q_start: pd.Timestamp) -> pd.Timestamp:
    # Origin is last day before the quarter start that exists in df
    origin = (q_start - pd.Timedelta(days=1)).normalize()
    # If missing, step backwards until we find a timestamp in df
    ts_set = set(pd.to_datetime(df_daily["timestamp"]).dt.normalize())
    while origin not in ts_set and origin > df_daily["timestamp"].min():
        origin -= pd.Timedelta(days=1)
    return origin


def forecast_to_date(
    model: DecoderOnlyTSModel,
    device: torch.device,
    series: np.ndarray,
    target_steps: int,
    p: int,
    *,
    anchor_to_last: bool = True,
) -> float:
    import math

    # Preserve the last real observation BEFORE any padding
    last_obs = float(series[-1]) if len(series) > 0 else float("nan")

    n_patches = math.ceil(len(series) / p)
    total = n_patches * p
    pad = total - len(series)
    if pad > 0:
        # Pad to full patch with constant last value instead of zeros to avoid bias
        # Note: anchoring uses last_obs explicitly, but constant pad also stabilizes normalization of later patches
        series = np.pad(series, (0, pad), mode="edge")
    patches = torch.from_numpy(series.reshape(n_patches, p)).unsqueeze(0).to(device)
    mean_s, std_s = scalar_stats(patches)
    patches_n = (patches - mean_s) / std_s
    yhat_path = model.generate(patches_n, steps=target_steps).detach().cpu().numpy()
    yhat_path = yhat_path * std_s + mean_s
    if anchor_to_last and np.isfinite(last_obs) and len(yhat_path) > 0:
        # Anchor to the actual last observed (pre-pad) value
        yhat_path = yhat_path + (last_obs - yhat_path[0])
    return float(yhat_path[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default="EUR")
    ap.add_argument("--quote", default="NOK")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--context_len", type=int, default=512)
    ap.add_argument("--anchor_to_last", type=str, default="true")
    ap.add_argument("--out", default="runs/quarterly_eval_eurnok_2000_2024")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = fetch_fx_pair(args.base, args.quote, args.start, args.end)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    model, dcfg, device = build_model(args.ckpt)
    p = int(dcfg.get("input_patch_len", 32))

    # Build daily view with ffill so quarter-end actuals are well-defined
    daily = (
        df.set_index("timestamp")["price"].asfreq("D").ffill().rename("price").reset_index()
    )
    daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True)

    start_ts = pd.to_datetime(args.start, utc=True)
    end_ts = pd.to_datetime(args.end, utc=True)
    q_bounds = quarter_boundaries(start_ts, end_ts)

    rows = []
    for q_start, q_end in q_bounds:
        origin = find_origin(daily, q_start)
        # context window
        ctx = daily[(daily["timestamp"] <= origin)].tail(args.context_len)
        if len(ctx) < max(16, p):
            continue
        # steps to quarter end
        steps = int((q_end - origin).days)
        if steps <= 0:
            continue
        y = ctx["price"].astype(np.float32).to_numpy()
        yhat_end = forecast_to_date(
            model,
            device,
            y,
            steps,
            p,
            anchor_to_last=(args.anchor_to_last.lower() in ("1", "true", "yes")),
        )
        # RW baseline: last observed value at origin
        rw_end = float(y[-1])
        # Actual at quarter end
        actual_end = float(daily[daily["timestamp"] == q_end]["price"].iloc[0]) if (daily["timestamp"] == q_end).any() else float("nan")
        rows.append({
            "quarter_start": q_start.date().isoformat(),
            "quarter_end": q_end.date().isoformat(),
            "origin": origin.date().isoformat(),
            "model": yhat_end,
            "rw": rw_end,
            "actual": actual_end,
            "abs_err_model": abs(yhat_end - actual_end) if np.isfinite(actual_end) else np.nan,
            "abs_err_rw": abs(rw_end - actual_end) if np.isfinite(actual_end) else np.nan,
        })

    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out, "quarterly_eval.csv")
    out_df.to_csv(out_csv, index=False)

    # Metrics
    valid = out_df[np.isfinite(out_df["abs_err_model"]) & np.isfinite(out_df["abs_err_rw"])].copy()
    mae_model = float(valid["abs_err_model"].mean()) if len(valid) else float("nan")
    mae_rw = float(valid["abs_err_rw"].mean()) if len(valid) else float("nan")
    gap = mae_rw - mae_model if np.isfinite(mae_rw) and np.isfinite(mae_model) else float("nan")
    dm_p = diebold_mariano((valid["abs_err_model"].to_numpy() - valid["abs_err_rw"].to_numpy()), h=1) if len(valid) else float("nan")
    with open(os.path.join(args.out, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"MAE_model: {mae_model}\n")
        f.write(f"MAE_rw: {mae_rw}\n")
        f.write(f"MAE_gap (rw - model): {gap}\n")
        f.write(f"DM_p (two-sided): {dm_p}\n")

    # Plot: quarter-end points
    hist = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].copy()
    hist = hist.sort_values("timestamp")
    # Build forecast_df compatible with plot_forecast (timestamps and forecast values)
    forecast_df = pd.DataFrame({
        "timestamp": pd.to_datetime(out_df["quarter_end"], utc=True),
        "forecast": out_df["model"],
    })
    rw_plot = pd.DataFrame({
        "timestamp": pd.to_datetime(out_df["quarter_end"], utc=True),
        "rw": out_df["rw"],
    })
    plot_forecast(
        history_df=hist,
        forecast_df=forecast_df,
        rw_df=rw_plot,
        title=f"{args.base}/{args.quote} Quarterly (cutoff day-before, {args.start}–{args.end})\nMAE model={mae_model:.4f} rw={mae_rw:.4f} gap={gap:.4f} DM p={dm_p:.3f}",
        save_path=os.path.join(args.out, "quarterly_eval.png"),
    )
    print(f"Saved quarterly eval CSV, plot, and metrics under {args.out}")


if __name__ == "__main__":
    main()
