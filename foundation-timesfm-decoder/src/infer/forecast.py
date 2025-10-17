from __future__ import annotations
import os
import argparse
import datetime as dt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from ..model.decoder_only import TimesFMDecoder
from ..model.layers import TransformerConfig
from ..utils.io import load_checkpoint, load_json, load_norm_stats


def infer_datetime_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if 'date' in c.lower() or 'time' in c.lower():
            return c
    # fallback: try parse first column
    c0 = df.columns[0]
    try:
        pd.to_datetime(df[c0])
        return c0
    except Exception:
        raise ValueError('Could not infer datetime column.')


def load_and_prepare(path: str):
    if path.lower().endswith('.parquet'):
        df = pd.read_parquet(path)
    elif path.lower().endswith('.csv'):
        df = pd.read_csv(path)
    else:
        raise ValueError('Unsupported input file format.')
    dt_col = infer_datetime_column(df)
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.set_index(dt_col).sort_index()
    # numeric only
    df_num = df.select_dtypes(include=[np.number])
    return df_num


def normalize(df_num: pd.DataFrame, mean: np.ndarray, std: np.ndarray):
    values = df_num.values.astype(np.float32)
    if values.shape[1] != mean.shape[0]:
        # project/univariate case: pad/truncate to match
        if values.shape[1] == 1:
            # map 1->in_dim by tiling the single series as a placeholder; model input_proj will learn mapping
            values = np.tile(values, (1, mean.shape[0]))
        else:
            # if fewer features, pad with zeros; if more, truncate
            if values.shape[1] < mean.shape[0]:
                pad = np.zeros((values.shape[0], mean.shape[0] - values.shape[1]), dtype=np.float32)
                values = np.concatenate([values, pad], axis=1)
            else:
                values = values[:, :mean.shape[0]]
    normed = (values - mean[None, :]) / (std[None, :] + 1e-6)
    return normed


def generate_forecast(model: TimesFMDecoder, series: np.ndarray, horizon: int = 90, context_len: int = 512, device='cpu'):
    # series: (N, C) normalized
    model.eval()
    C = series.shape[1]
    outputs = []
    latents = []
    with torch.no_grad():
        ctx = series[-context_len:] if len(series) >= context_len else np.pad(series, ((context_len-len(series),0),(0,0)))
        x = torch.from_numpy(ctx).float().unsqueeze(0).to(device)  # (1, T, C)
        for t in range(horizon):
            y, h = model(x)
            latents.append(h[:, -1:].detach().cpu())
            next_step = y[:, -1:]
            x = torch.cat([x[:, 1:], next_step], dim=1)
            outputs.append(next_step.detach().cpu())
    yhat = torch.cat(outputs, dim=1).squeeze(0).numpy()  # (H, C)
    H = yhat.shape[0]
    # simple CI via MC dropout: enable dropout at test time
    model.train()
    mc_samples = []
    S = 20
    with torch.no_grad():
        x = torch.from_numpy(ctx).float().unsqueeze(0).to(device)
        for s in range(S):
            outs = []
            x_s = x.clone()
            for t in range(horizon):
                y, _ = model(x_s)
                next_step = y[:, -1:]
                x_s = torch.cat([x_s[:, 1:], next_step], dim=1)
                outs.append(next_step.detach().cpu())
            mc_samples.append(torch.cat(outs, dim=1).squeeze(0))
    mc = torch.stack(mc_samples, dim=0).numpy()  # (S, H, C)
    lower = np.percentile(mc, 5, axis=0)
    upper = np.percentile(mc, 95, axis=0)

    latents = torch.cat(latents, dim=1).squeeze(0).numpy()  # (H, T=1, d)->(H, d) via squeeze later

    return yhat, lower, upper, latents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_PATH', dest='data_path', type=str, required=False)
    parser.add_argument('--H', dest='h', type=int, default=90)
    args, unknown = parser.parse_known_args()

    cfg_path = os.environ.get('CFG', 'configs/default.yaml')
    import yaml
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load stats
    mean, std = load_norm_stats(cfg['paths']['norm_stats'])

    # load model
    model_cfg = TransformerConfig(
        d_model=cfg['model']['d_model'],
        n_heads=cfg['model']['n_heads'],
        n_layers=cfg['model']['n_layers'],
        ffn_dim=cfg['model']['ffn_dim'],
        dropout=cfg['model']['dropout'],
        context_len=cfg['model']['context_len'],
    )
    model = TimesFMDecoder(in_dim=cfg['model']['in_dim'], config=model_cfg).to(device)

    ckpt_path = os.path.join(cfg['paths']['checkpoints_dir'], 'foundation_decoder.pt')
    state_dict, _, extra = load_checkpoint(ckpt_path, map_location=device)
    if state_dict is None:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Train the model first.")
    model.load_state_dict(state_dict)

    # data
    if args.data_path is None:
        raise ValueError('Please provide DATA_PATH to a csv/parquet file.')
    df = load_and_prepare(args.data_path)
    normed = normalize(df, mean, std)

    yhat, lower, upper, latents = generate_forecast(model, normed, horizon=args.h, context_len=cfg['model']['context_len'], device=device)

    # denormalize predictions based on primary column (first feature)
    pred_primary = yhat[:, 0] * (std[0] + 1e-6) + mean[0]
    low_primary = lower[:, 0] * (std[0] + 1e-6) + mean[0]
    up_primary = upper[:, 0] * (std[0] + 1e-6) + mean[0]

    # build timestamps
    last_ts = df.index[-1]
    freq = pd.infer_freq(df.index)
    if freq is None:
        # assume daily
        freq = 'D'
    future_index = pd.date_range(start=last_ts, periods=args.h+1, freq=freq)[1:]

    out = pd.DataFrame({
        'timestamp': future_index,
        'prediction': pred_primary,
        'lower_ci': low_primary,
        'upper_ci': up_primary,
    })

    os.makedirs(cfg['paths']['outputs_dir'], exist_ok=True)
    out_path = os.path.join(cfg['paths']['outputs_dir'], f"forecast_{dt.datetime.now().strftime('%Y%m%d')}.csv")
    out.to_csv(out_path, index=False)

    # save latents
    np.save(os.path.join(cfg['paths']['outputs_dir'], f"latents_{dt.datetime.now().strftime('%Y%m%d')}.npy"), latents)

    print(f"Saved forecast to {out_path}")


if __name__ == '__main__':
    main()
