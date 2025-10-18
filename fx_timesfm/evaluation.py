from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
try:  # optional SciPy for Student-t CDF
    from scipy.stats import t as student_t  # type: ignore
except Exception:  # pragma: no cover - fallback when SciPy absent
    student_t = None  # type: ignore

from .metrics import mae, msmape
from .resample import to_monthly, to_quarterly
import torch
from torch import Tensor
from typing import Optional
import numpy as np


def rw_forecast_last(series: pd.Series, horizon_steps: int = 1) -> pd.Series:
    # Random-walk: last observed level at origin; for horizon=1 step monthly/quarterly, predict last value
    # For evaluation alignment, shift by 1
    return series.shift(1)


def diebold_mariano(diff_losses: np.ndarray, h: int, nw_lag: int | None = None) -> float:
    """Two-sided Diebold–Mariano test with Newey–West correction.

    Parameters
    ----------
    diff_losses : np.ndarray
        Loss differential series, e.g., |e_model| - |e_rw| (length T).
    h : int
        Forecast horizon in steps; sets NW bandwidth to h-1 (non-overlapping when h=1).

    Returns
    -------
    float
        Two-sided p-value under t-approximation.
    """
    d = np.asarray(diff_losses, dtype=float)
    T = d.size
    if T < 5 or not np.all(np.isfinite(d)):
        return 1.0
    d_bar = float(d.mean())
    # Newey–West variance of the sample mean using demeaned series
    e = d - d_bar
    # Default NW bandwidth is h-1 (non-overlapping forecasts); allow override
    lag = max(0, int(nw_lag)) if nw_lag is not None else max(0, h - 1)
    gamma0 = float((e @ e) / T)
    var = gamma0
    for l in range(1, lag + 1):
        cov = float((e[l:] @ e[:-l]) / T)
        w = 1.0 - l / (lag + 1)
        var += 2.0 * w * cov
    var = var / T  # variance of the mean
    if not np.isfinite(var) or var <= 0.0:
        return 1.0
    dm_stat = d_bar / math.sqrt(var)
    # t-approximation with optional SciPy; fallback to normal if SciPy missing
    if student_t is not None:
        cdf_val = float(student_t.cdf(abs(dm_stat), df=T - 1))  # type: ignore[attr-defined]
    else:
        # Normal approximation
        cdf_val = 0.5 * (1.0 + math.erf(abs(dm_stat) / math.sqrt(2.0)))
    p = 2.0 * (1.0 - cdf_val)
    return float(p)


def evaluate_monthly_quarterly(df_long: pd.DataFrame, symbol_col: str = "symbol", timestamp_col: str = "timestamp", value_col: str = "price") -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {"monthly": {}, "quarterly": {}}
    for freq_name, to_func, h in [("monthly", to_monthly, 1), ("quarterly", to_quarterly, 1)]:
        maes: List[float] = []
        msms: List[float] = []
        dm_ps: List[float] = []
        for sym, g in df_long.groupby(symbol_col):
            s = to_func(g[[timestamp_col, value_col]].rename(columns={timestamp_col: "timestamp", value_col: "price"}))
            s = s.dropna()
            if len(s) < 10:
                continue
            y = s["price"].reset_index(drop=True)
            # RW baseline
            y_hat_rw = rw_forecast_last(y, horizon_steps=h)
            # Align
            mask = y_hat_rw.notna()
            y1 = y[mask]
            rw1 = y_hat_rw[mask]
            e_rw = np.abs(rw1.values - y1.values)
            # Store baseline MAE for later comparison
            maes.append(float(np.mean(e_rw)))
            msms.append(float(np.mean(np.abs(rw1.values - y1.values) / np.clip(np.abs(y1.values), 1e-3, None))))
            # For DM test, diff_losses should be l_model - l_rw; here use model omitted; caller can replace
            # Provide DM p-value baseline vs itself as 1.0
            dm_ps.append(1.0)
        results[freq_name] = {
            "mae": float(np.mean(maes)) if maes else float("nan"),
            "msmape": float(np.mean(msms)) if msms else float("nan"),
            "dm_p": float(np.mean(dm_ps)) if dm_ps else float("nan"),
        }
    return results


@torch.no_grad()
def evaluate_model_vs_rw(
    model,
    df_long: pd.DataFrame,
    p: int,
    h: int,
    context_len: int,
    device: Optional[torch.device] = None,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    value_col: str = "price",
) -> Dict[str, Dict[str, float]]:
    """Evaluate model vs RW at monthly and quarterly with DM tests.

    For each symbol:
      - Resample to monthly/quarterly end-of-period values
      - Roll through time, each origin t forms a context (up to context_len), model predicts 1-step ahead (first element of last token's output patch)
      - Compare to RW (last observed)
      - Compute MAE, msMAPE and DM p-values (model vs RW)
    Returns macro-averaged metrics across symbols.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    def series_to_forecasts(s: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        y_vals = s.values.astype(np.float32)
        preds = []
        rws = []
        for t in range(1, len(y_vals)):
            start = max(0, t - context_len)
            ctx = y_vals[start:t]
            if len(ctx) < p:
                continue
            # patchify
            n_patches = int(np.ceil(len(ctx) / p))
            total = n_patches * p
            pad = total - len(ctx)
            if pad > 0:
                ctx_pad = np.pad(ctx, (0, pad))
            else:
                ctx_pad = ctx
            patches = torch.from_numpy(ctx_pad.reshape(n_patches, p)).unsqueeze(0).to(device)
            attn_pad = torch.zeros((1, patches.size(1)), dtype=torch.bool, device=device)
            # simple first-patch stats
            mean = patches[:, :1, :].mean(dim=(1, 2), keepdim=True)
            std = patches[:, :1, :].std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            patches_n = (patches - mean) / std
            out = model(patches_n, attn_pad, None)["pred"]  # [1, T, h]
            next_patch = out[:, -1, :]  # [1, h]
            yhat_n = next_patch[0, 0].item()  # take first step of horizon
            yhat = yhat_n * std.view(-1).item() + mean.view(-1).item()
            preds.append(yhat)
            rws.append(y_vals[t - 1])
        return np.asarray(preds), np.asarray(y_vals[len(y_vals) - len(preds):])

    results: Dict[str, Dict[str, float]] = {}
    for label, to_func in [("monthly", to_monthly), ("quarterly", to_quarterly)]:
        maes_sym: List[float] = []
        msms_sym: List[float] = []
        gaps_sym: List[float] = []
        dm_ps: List[float] = []
        for sym, g in df_long.groupby(symbol_col):
            s = to_func(g[[timestamp_col, value_col]].rename(columns={timestamp_col: "timestamp", value_col: "price"}))
            s = s.dropna()
            if len(s) < (p + 5):
                continue
            y = s["price"].reset_index(drop=True)
            yhat, y_true = series_to_forecasts(y)
            if len(yhat) == 0:
                continue
            # Align RW
            rw = y.shift(1).dropna().values[-len(yhat):]
            e_model = np.abs(yhat - y_true)
            e_rw = np.abs(rw - y_true)
            maes_sym.append(float(e_model.mean()))
            msms_sym.append(float(np.mean(np.abs(yhat - y_true) / np.clip(np.abs(y_true), 1e-3, None))))
            gaps_sym.append(float(e_rw.mean() - e_model.mean()))
            dm_ps.append(diebold_mariano(e_model - e_rw, h=1))
        results[label] = {
            "model_mae": float(np.mean(maes_sym)) if maes_sym else float("nan"),
            "msmape": float(np.mean(msms_sym)) if msms_sym else float("nan"),
            "mae_gap_vs_rw": float(np.mean(gaps_sym)) if gaps_sym else float("nan"),
            "dm_p": float(np.mean(dm_ps)) if dm_ps else float("nan"),
        }
    return results
