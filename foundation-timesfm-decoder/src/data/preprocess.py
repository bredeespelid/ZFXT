from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ..utils.io import save_norm_stats


def load_dataset(path: str) -> pd.DataFrame:
    if path.lower().endswith('.parquet'):
        df = pd.read_parquet(path)
    elif path.lower().endswith('.csv'):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    # ensure datetime index
    if 'date' in df.columns:
        dt_col = 'date'
    else:
        # try first datetime-like column
        dt_candidates = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
        dt_col = dt_candidates[0] if dt_candidates else None
    if dt_col is not None:
        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.set_index(dt_col)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError('Dataset must have or infer a datetime index.')
    df = df.sort_index()
    # keep only numeric
    df = df.select_dtypes(include=[np.number])
    return df


def normalize_and_split(df: pd.DataFrame, norm_stats_path: str, val_split: float = 0.1, seed: int = 1337):
    values = df.values.astype(np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True) + 1e-6
    normed = (values - mean) / std
    save_norm_stats(norm_stats_path, mean.squeeze(0), std.squeeze(0))

    # split by time: last val_split fraction as validation
    n = len(df)
    n_val = int(n * val_split)
    train_arr = normed[:-n_val] if n_val > 0 else normed
    val_arr = normed[-n_val:] if n_val > 0 else normed[-1:]
    train_index = df.index[:-n_val] if n_val > 0 else df.index
    val_index = df.index[-n_val:] if n_val > 0 else df.index[-1:]

    return (train_arr, train_index), (val_arr, val_index), mean.squeeze(0), std.squeeze(0)
