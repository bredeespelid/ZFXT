from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


# ----------------------
# Data structures
# ----------------------


@dataclass
class DataConfig:
    train_paths: List[str]
    val_paths: List[str]
    test_paths: List[str]
    symbols: List[str]
    timestamp_col: str
    symbol_col: str
    price_col: str
    freq: str
    resample: bool
    fill_method: str
    context_len: int
    input_patch_len: int
    output_patch_len: int
    horizon: int
    max_context_patches: int
    batch_size: int
    num_workers: int


def _read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {ext}")


def load_price_frames(paths: List[str], timestamp_col: str, symbol_col: str, price_col: str) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not os.path.exists(p):
            continue
        df = _read_any(p)
        if timestamp_col not in df.columns:
            raise KeyError(f"missing {timestamp_col} in {p}")
        # infer symbol if absent
        if symbol_col not in df.columns:
            df[symbol_col] = os.path.splitext(os.path.basename(p))[0]
        cols = [timestamp_col, symbol_col, price_col]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"missing columns {missing} in {p}")
        frames.append(df[cols].copy())
    if not frames:
        raise FileNotFoundError("No input data files found.")
    df = pd.concat(frames, ignore_index=True)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    df = df.sort_values([symbol_col, timestamp_col])
    return df


def resample_uniform(
    df: pd.DataFrame,
    freq: str,
    timestamp_col: str,
    symbol_col: str,
    price_col: str,
    fill_method: str = "ffill",
) -> pd.DataFrame:
    # Resample per symbol to uniform cadence using pandas asfreq + fill
    out = []
    for sym, g in df.groupby(symbol_col):
        g = g.set_index(timestamp_col).sort_index()
        g = g[[price_col]]
        g = g.asfreq(freq)
        if fill_method == "ffill":
            g = g.ffill()
        elif fill_method == "bfill":
            g = g.bfill()
        elif fill_method == "interpolate":
            g = g.interpolate(limit_direction="both")
        g[symbol_col] = sym
        out.append(g.reset_index())
    df2 = pd.concat(out, ignore_index=True)
    return df2[[timestamp_col, symbol_col, price_col]].sort_values([symbol_col, timestamp_col])


def rolling_windows(series: np.ndarray, window: int, step: int = 1) -> Iterator[np.ndarray]:
    n = len(series)
    i = 0
    while i + window <= n:
        yield series[i : i + window]
        i += step


def patchify(x: np.ndarray, patch_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split 1D array into non-overlapping patches of length patch_len.

    Returns (patches, pad_mask), where pad_mask is 1 where padding was applied inside the last patch.
    """
    n = len(x)
    n_patches = math.ceil(n / patch_len)
    total = n_patches * patch_len
    pad = total - n
    if pad > 0:
        x_pad = np.pad(x, (0, pad), mode="constant")
        pad_mask = np.zeros(total, dtype=np.float32)
        pad_mask[-pad:] = 1.0
    else:
        x_pad = x
        pad_mask = np.zeros(total, dtype=np.float32)
    patches = x_pad.reshape(n_patches, patch_len)
    pad_mask = pad_mask.reshape(n_patches, patch_len)
    return patches.astype(np.float32), pad_mask


class FXDataset(Dataset):
    """Global univariate dataset with rolling windows per symbol.

    Each item returns a single window consisting of up to context_len points.
    """

    def __init__(
        self,
        frames: pd.DataFrame,
        symbols: Optional[List[str]],
        cfg: DataConfig,
        split: str,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.timestamp_col = cfg.timestamp_col
        self.symbol_col = cfg.symbol_col
        self.price_col = cfg.price_col
        self.context_len = cfg.context_len
        self.input_patch_len = cfg.input_patch_len
        self.output_patch_len = cfg.output_patch_len
        self.horizon = cfg.horizon
        self.max_context_patches = cfg.max_context_patches

        if symbols:
            frames = frames[frames[self.symbol_col].isin(symbols)]
        self.symbol_groups: Dict[str, np.ndarray] = {}
        for sym, g in frames.groupby(self.symbol_col):
            self.symbol_groups[sym] = g[self.price_col].to_numpy(dtype=np.float32)

        # Precompute indices for rolling windows per symbol
        self.index: List[Tuple[str, int]] = []
        L = self.context_len
        step = self.output_patch_len  # stride windows by horizon for efficiency
        for sym, arr in self.symbol_groups.items():
            n = len(arr)
            # ensure room for at least context
            i = 0
            while i + L <= n:
                self.index.append((sym, i))
                i += step

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sym, i = self.index[idx]
        arr = self.symbol_groups[sym]
        L = self.context_len
        x = arr[i : i + L]
        patches, pad_mask = patchify(x, self.input_patch_len)
        # cap number of patches (e.g., 512/32=16)
        if patches.shape[0] > self.max_context_patches:
            patches = patches[-self.max_context_patches :]
            pad_mask = pad_mask[-self.max_context_patches :]
        return {
            "symbol": sym,
            "patches": torch.from_numpy(patches),  # [T_p, p]
            "pad_mask": torch.from_numpy(pad_mask),  # [T_p, p]
        }


def collate_batch(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Top-level collate function (picklable for Windows multiprocessing)."""
    max_Tp = max(item["patches"].shape[0] for item in batch)
    p = batch[0]["patches"].shape[1]
    patches_list: List[Tensor] = []
    pad_mask_list: List[Tensor] = []
    attn_pad: List[Tensor] = []
    for item in batch:
        Tpi = item["patches"].shape[0]
        pad_T = max_Tp - Tpi
        if pad_T > 0:
            patches_pad = torch.zeros((pad_T, p), dtype=item["patches"].dtype)
            mask_pad = torch.ones((pad_T, p), dtype=item["pad_mask"].dtype)
            patches_list.append(torch.cat([patches_pad, item["patches"]], dim=0))
            pad_mask_list.append(torch.cat([mask_pad, item["pad_mask"]], dim=0))
        else:
            patches_list.append(item["patches"])  # already max
            pad_mask_list.append(item["pad_mask"])  # already max
        # attention pad: 1 for padded tokens
        attn_pad.append(torch.cat([torch.ones(pad_T), torch.zeros(Tpi)]))

    return {
        "patches": torch.stack(patches_list, dim=0),  # [B, T_p, p]
        "patch_pad_mask": torch.stack(pad_mask_list, dim=0),  # [B, T_p, p]
        "attn_pad_mask": torch.stack(attn_pad, dim=0).bool(),  # [B, T_p]
    }


def make_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader]:
    df_train = load_price_frames(cfg.train_paths, cfg.timestamp_col, cfg.symbol_col, cfg.price_col)
    if cfg.resample:
        df_train = resample_uniform(df_train, cfg.freq, cfg.timestamp_col, cfg.symbol_col, cfg.price_col, cfg.fill_method)
    df_val = load_price_frames(cfg.val_paths, cfg.timestamp_col, cfg.symbol_col, cfg.price_col)
    if cfg.resample:
        df_val = resample_uniform(df_val, cfg.freq, cfg.timestamp_col, cfg.symbol_col, cfg.price_col, cfg.fill_method)

    train_ds = FXDataset(df_train, cfg.symbols or None, cfg, split="train")
    val_ds = FXDataset(df_val, cfg.symbols or None, cfg, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_batch,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_batch,
        persistent_workers=False,
    )
    return train_loader, val_loader
