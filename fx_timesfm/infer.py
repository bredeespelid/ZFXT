from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from .model import DecoderOnlyTSModel
from .checkpoints import load_ckpt
from .model_card import ModelCard
from .resample import to_freq
from .plot_utils import plot_forecast


def load_context_from_csv(path: str, value_col: str = "price") -> np.ndarray:
    df = pd.read_csv(path)
    if value_col not in df.columns:
        # take the last numeric column
        num_cols = [c for c in df.columns if df[c].dtype != object]
        if not num_cols:
            raise ValueError("No numeric column found for context")
        value_col = num_cols[-1]
    arr = df[value_col].to_numpy(dtype=np.float32)
    return arr


def patchify_np(x: np.ndarray, patch_len: int) -> np.ndarray:
    n = len(x)
    n_patches = int(np.ceil(n / patch_len))
    total = n_patches * patch_len
    pad = total - n
    if pad > 0:
        # Pad with the last value to preserve level and avoid zero-padding artifacts
        x = np.pad(x, (0, pad), mode="edge")
    return x.reshape(n_patches, patch_len)


def forecast_from_df(
    df: pd.DataFrame,
    ckpt_path: str,
    horizon: int,
    *,
    context_len: Optional[int] = None,
    anchor_to_last: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Load model and card
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
    model.to(device)
    model.eval()

    # Frequency handling
    freq = card.train_freq if card else dcfg.get("freq", "D")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    # Auto-resample to the ModelCard training frequency
    try:
        from .resample import to_freq as _to_freq
        df = _to_freq(df, freq=freq, timestamp_col="timestamp", value_col="price", method="ffill")
    except Exception as e:
        print(f"Resample warning: {e}")

    # Build context from tail of series (recent window), not the entire history
    p = int(dcfg.get("input_patch_len", 32))
    max_ctx_default = int(dcfg.get("context_len", 512))
    ctx_len = int(context_len) if context_len is not None else max_ctx_default
    series_full = df["price"].to_numpy(dtype=np.float32)
    series = series_full[-ctx_len:]
    patches = patchify_np(series, p)
    # cap number of context patches if needed
    max_context_patches = int(dcfg.get("max_context_patches", 16))
    if patches.shape[0] > max_context_patches:
        patches = patches[-max_context_patches:]
    patches_t = torch.from_numpy(patches).unsqueeze(0)
    mean = patches_t[:, :1, :].mean(dim=(1, 2), keepdim=True)
    std = patches_t[:, :1, :].std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    patches_n = (patches_t - mean) / std
    preds_n = model.generate(patches_n.to(device), steps=horizon)
    # Denormalize with scalar first-patch stats to avoid patch-dimension broadcasting artifacts
    mean_s = float(mean.view(-1)[0].item())
    std_s = float(std.view(-1)[0].item())
    preds = (preds_n.cpu().numpy() * std_s) + mean_s
    if anchor_to_last and len(series) > 0 and len(preds) > 0:
        delta = float(series[-1]) - float(preds[0])
        preds = preds + delta

    # Build timestamps for horizon by extending last timestamp with target freq
    last_ts = df["timestamp"].iloc[-1]
    idx = pd.date_range(start=last_ts, periods=horizon + 1, freq=freq)[1:]
    forecast_df = pd.DataFrame({"timestamp": idx, "forecast": preds})

    # RW baseline: repeat last observed level
    rw_df = pd.DataFrame({"timestamp": idx, "rw": [series[-1] if len(series) > 0 else np.nan] * horizon})
    return forecast_df, rw_df


def run_example_from_norgesbank(
    ckpt_path: str,
    horizon: int = 128,
    *,
    anchor_to_last: bool = True,
    context_len: Optional[int] = None,
    show_actuals: bool = False,
) -> None:
    from fx_timesfm.examples.fetch_norgesbank_exr import fetch_and_parse
    df = fetch_and_parse()
    df = df.sort_values("timestamp")
    print(f"Fetched {len(df)} daily EUR/NOK observations from Norges Bank.")
    # Ensure checkpoint exists or try fallback
    if not os.path.exists(ckpt_path):
        fallback = os.path.join("checkpoints", "model_best.pt")
        if os.path.exists(fallback):
            print(f"Checkpoint not found at {ckpt_path}. Falling back to {fallback}")
            ckpt_path = fallback
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train a model or point to a valid ckpt.")
    forecast_df, rw_df = forecast_from_df(df, ckpt_path, horizon, context_len=context_len, anchor_to_last=anchor_to_last)
    actuals_df = None
    # Optionally overlay actuals beyond end date (e.g., fetch next year) for visual check
    if show_actuals:
        try:
            from fx_timesfm.examples.fetch_norgesbank_exr import fetch_range
            start_next = (df["timestamp"].max() + pd.Timedelta(days=1)).date().isoformat()
            end_next = (df["timestamp"].max() + pd.Timedelta(days=horizon + 2)).date().isoformat()
            df_next = fetch_range(start=start_next, end=end_next)
            actuals_df = df_next.sort_values("timestamp")
            df = pd.concat([df, df_next]).drop_duplicates("timestamp").sort_values("timestamp")
        except Exception as e:
            print(f"Could not fetch actuals overlay: {e}")
    os.makedirs("runs/plots", exist_ok=True)
    plot_forecast(
        df,
        forecast_df,
        rw_df,
        title=f"EUR/NOK Forecast {df['timestamp'].min().date()}–{df['timestamp'].max().date()}",
        save_path="runs/plots/eurnok_demo.png",
        actuals_df=actuals_df,
    )
    out_csv = "runs/plots/eurnok_demo_forecast.csv"
    out_df = forecast_df.rename(columns={"forecast": "model"}).set_index("timestamp")
    out_df = out_df.join(rw_df.set_index("timestamp"), how="left")
    if actuals_df is not None:
        out_df = out_df.join(actuals_df.set_index("timestamp").rename(columns={"price": "actual"}), how="left")
    out_df.reset_index().to_csv(out_csv, index=False)
    print(f"Saved forecast plot and CSV under runs/plots/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=256)
    parser.add_argument("--context_csv", type=str, required=False, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--context_len", type=int, default=None)
    parser.add_argument("--anchor_to_last", type=str, default="false")
    parser.add_argument("--demo_norgesbank", type=str, default="false")
    parser.add_argument("--show_actuals", type=str, default="false")
    args = parser.parse_args()

    if args.demo_norgesbank.lower() in ("1", "true", "yes"):
        run_example_from_norgesbank(
            args.ckpt,
            horizon=int(args.horizon),
            anchor_to_last=(args.anchor_to_last.lower() in ("1", "true", "yes")) if args.anchor_to_last is not None else True,
            context_len=int(args.context_len) if args.context_len is not None else None,
            show_actuals=(args.show_actuals.lower() in ("1", "true", "yes")) if args.show_actuals is not None else False,
        )
        return
    if not args.context_csv:
        raise SystemExit("--context_csv is required unless --demo_norgesbank true")
    payload, card = load_ckpt(args.ckpt)
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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    series = load_context_from_csv(args.context_csv)
    patches = patchify_np(series, dcfg.get("input_patch_len", 32))
    patches_t = torch.from_numpy(patches).unsqueeze(0)  # [1, T, p]
    # simple normalization using first patch stats
    first = patches_t[:, :1, :]
    mean = first.mean(dim=(1, 2), keepdim=True)
    std = first.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    patches_norm = (patches_t - mean) / std

    preds_norm = model.generate(patches_norm.to(device), steps=int(args.horizon))
    mean_s = float(mean.view(-1)[0].item())
    std_s = float(std.view(-1)[0].item())
    preds = preds_norm.cpu() * std_s + mean_s
    # Optional anchoring in CLI path too
    if args.anchor_to_last and str(args.anchor_to_last).lower() in ("1", "true", "yes") and len(series) > 0:
        delta = float(series[-1]) - float(preds[0].item())
        preds = preds + delta

    out_csv = os.path.splitext(args.ckpt)[0] + f"_forecast_{args.horizon}.csv"
    pd.DataFrame({"forecast": preds.numpy()}).to_csv(out_csv, index=False)
    meta = {
        "ckpt": args.ckpt,
        "horizon": int(args.horizon),
        "context_len": len(series),
        "config_hash": payload.get("cfg_hash", ""),
        "model_card": card.to_json() if card else None,
    }
    with open(os.path.splitext(out_csv)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved forecast to {out_csv}")


if __name__ == "__main__":
    main()
