from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch

from fx_timesfm.data_sources.norgesbank import fetch_fx_pair
from fx_timesfm.checkpoints import load_ckpt
from fx_timesfm.model import DecoderOnlyTSModel
from fx_timesfm.resample import to_freq
from fx_timesfm.plot_utils import plot_forecast


def build_model_from_ckpt(ckpt: str) -> tuple[DecoderOnlyTSModel, dict, dict]:
    payload, card = load_ckpt(ckpt)
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
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, dcfg, {"device": device, "card": card}


def scalar_norm_stats(patches: torch.Tensor) -> tuple[float, float]:
    mean = patches[:, :1, :].mean(dim=(1, 2), keepdim=True)
    std = patches[:, :1, :].std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return float(mean.view(-1)[0].item()), float(std.view(-1)[0].item())


def walk_forward(
    df: pd.DataFrame,
    model: DecoderOnlyTSModel,
    dcfg: dict,
    *,
    start_date: str,
    end_date: str,
    freq: str = "ME",
    context_len: int = 512,
    anchor_to_last: bool = True,
) -> pd.DataFrame:
    # Resample and slice
    dfr = to_freq(df.rename(columns={"timestamp": "timestamp", "price": "price"}), freq=freq)
    # Ensure tz-aware timestamps (UTC) for consistent comparisons
    dfr["timestamp"] = pd.to_datetime(dfr["timestamp"], utc=True, errors="coerce")
    start_ts = pd.to_datetime(start_date, utc=True)
    end_ts = pd.to_datetime(end_date, utc=True)
    dfr = dfr[(dfr["timestamp"] >= start_ts) & (dfr["timestamp"] <= end_ts)]
    dfr = dfr.sort_values("timestamp").reset_index(drop=True)
    # Cap context length to available data
    if len(dfr) < context_len + 2:
        context_len = max(8, len(dfr) - 2)

    p = int(dcfg.get("input_patch_len", 32))
    h = int(dcfg.get("output_patch_len", 128))
    device = next(model.parameters()).device

    preds = []
    rws = []
    stamps = []
    series = dfr["price"].astype(np.float32).to_numpy()

    import math
    for t in range(context_len, len(series)):
        ctx = series[t - context_len : t]
        # patchify
        n_patches = math.ceil(len(ctx) / p)
        total = n_patches * p
        pad = total - len(ctx)
        if pad > 0:
            # Pad with the last observed value instead of zeros to avoid distorting normalization
            ctx = np.pad(ctx, (0, pad), mode="edge")
        patches = torch.from_numpy(ctx.reshape(n_patches, p)).unsqueeze(0).to(device)
        mean_s, std_s = scalar_norm_stats(patches)
        patches_n = (patches - mean_s) / std_s
        yhat = model.generate(patches_n, steps=1).cpu().numpy()[0] * std_s + mean_s
        if anchor_to_last:
            yhat = yhat + (series[t - 1] - yhat)
        preds.append(float(yhat))
        rws.append(float(series[t - 1]))
        stamps.append(dfr["timestamp"].iloc[t])

    out = pd.DataFrame({"timestamp": stamps, "model": preds, "rw": rws})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default="EUR")
    ap.add_argument("--quote", default="NOK")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--freq", default="ME", help="Target resample frequency (ME=month-end, QE=quarter-end)")
    ap.add_argument("--context_len", type=int, default=512)
    ap.add_argument("--anchor_to_last", type=str, default="true")
    ap.add_argument("--out", default="runs/walk_forward_eurnok_2000_2024")
    args = ap.parse_args()

    df = fetch_fx_pair(args.base, args.quote, args.start, args.end)
    model, dcfg, env = build_model_from_ckpt(args.ckpt)

    res = walk_forward(
        df,
        model,
        dcfg,
        start_date=args.start,
        end_date=args.end,
        freq=args.freq,
        context_len=int(args.context_len),
        anchor_to_last=(args.anchor_to_last.lower() in ("1", "true", "yes")),
    )

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "walk_forward.csv")
    res.to_csv(out_csv, index=False)

    # Build a plot: combine history and forecasts, overlay RW baseline
    hist = df.copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True, errors="coerce")
    start_ts = pd.to_datetime(args.start, utc=True)
    end_ts = pd.to_datetime(args.end, utc=True)
    hist = hist[(hist["timestamp"] >= start_ts) & (hist["timestamp"] <= end_ts)]
    hist = hist.sort_values("timestamp").reset_index(drop=True)

    plot_forecast(
        history_df=hist,
        forecast_df=res.rename(columns={"model": "forecast"}),
        rw_df=res[["timestamp", "rw"]],
        title=f"{args.base}/{args.quote} Walk-Forward {args.start}–{args.end}",
        save_path=os.path.join(args.out, "walk_forward.png"),
    )
    print(f"Saved walk-forward CSV and plot under {args.out}")


if __name__ == "__main__":
    main()
