from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from fx_timesfm.checkpoints import load_ckpt
from fx_timesfm.model import DecoderOnlyTSModel
from fx_timesfm.plot_utils import plot_forecast


def build_model_from_ckpt(ckpt: str) -> tuple[DecoderOnlyTSModel, dict, torch.device]:
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, dcfg, device


def scalar_stats(patches: torch.Tensor) -> tuple[float, float]:
    mean = patches[:, :1, :].mean(dim=(1, 2), keepdim=True)
    std = patches[:, :1, :].std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return float(mean.view(-1)[0].item()), float(std.view(-1)[0].item())


def patchify_edgepad(x: np.ndarray, p: int) -> np.ndarray:
    n = len(x)
    n_patches = int(np.ceil(n / p))
    total = n_patches * p
    pad = total - n
    if pad > 0:
        x = np.pad(x, (0, pad), mode="edge")
    return x.reshape(n_patches, p)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _match_col(df: pd.DataFrame, target: str, alternatives: Optional[List[str]] = None) -> str:
    names = {c.lower(): c for c in df.columns}
    cand = [target.lower()] + [a.lower() for a in (alternatives or [])]
    for k in cand:
        if k in names:
            return names[k]
    raise KeyError(f"Required column not found: {target} (have: {list(df.columns)})")


def _compute_base_walk_forward(
    y: np.ndarray,
    model: DecoderOnlyTSModel,
    p: int,
    context_len: int,
    anchor_to_last: bool,
) -> np.ndarray:
    device = next(model.parameters()).device
    yhat = np.full_like(y, np.nan, dtype=np.float32)
    for t in range(1, len(y)):
        start = max(0, t - context_len)
        y_ctx = y[start:t]
        if len(y_ctx) < p:
            continue
        patches = torch.from_numpy(patchify_edgepad(y_ctx, p)).unsqueeze(0).to(device)
        mean_s, std_s = scalar_stats(patches)
        patches_n = (patches - mean_s) / std_s
        with torch.no_grad():
            out = model(patches_n, attn_pad_mask=torch.zeros((1, patches.size(1)), dtype=torch.bool, device=device))
            yhat_n = out["pred"][:, -1, 0].item()
        y1 = yhat_n * std_s + mean_s
        if anchor_to_last and len(y_ctx) > 0:
            y1 = y1 + (y_ctx[-1] - y1)
        yhat[t] = float(y1)
    return yhat


def walk_forward_with_covariates(
    df: pd.DataFrame,
    model: DecoderOnlyTSModel,
    dcfg: dict,
    *,
    date_col: str = "DATE",
    price_col: str = "EUR_NOK",
    covariate_cols: Optional[List[str]] = None,
    context_len: int = 512,
    anchor_to_last: bool = True,
    ols_window: int = 128,
) -> pd.DataFrame:
    covariate_cols = covariate_cols or []

    # Prepare series
    d = _normalize_columns(df)
    # Map columns case-insensitively
    date_col = _match_col(d, date_col, ["date", "Timestamp", "timestamp"])  # type: ignore[arg-type]
    price_col = _match_col(d, price_col, ["EUR/NOK", "eurnok", "price"])  # type: ignore[arg-type]
    covariate_cols = [
        _match_col(d, c, [c.replace("_", "").replace("/", "")]) if c not in d.columns else c for c in covariate_cols
    ]

    d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, utc=True, errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    y = d[price_col].astype(np.float32).to_numpy()

    # Where available, forward-fill covariates and standardize within context windows
    X = None
    if covariate_cols:
        X = d[covariate_cols].astype(np.float32).to_numpy()

    p = int(dcfg.get("input_patch_len", 32))
    h = int(dcfg.get("output_patch_len", 128))
    device = next(model.parameters()).device

    stamps: List[pd.Timestamp] = []
    preds: List[float] = []
    preds_base: List[float] = []

    # Precompute base one-step walk-forward predictions (model only)
    base_preds_all = _compute_base_walk_forward(y, model, p, context_len, anchor_to_last)

    # Walk forward one day at a time with covariate adjustment
    for t in range(1, len(y)):
        start = max(0, t - context_len)
        y_ctx = y[start:t]
        if len(y_ctx) < p:
            continue

        yhat_base = float(base_preds_all[t]) if np.isfinite(base_preds_all[t]) else float("nan")
        if not np.isfinite(yhat_base):
            continue

        # Covariate adjustment via rolling OLS on residuals: r_t = y_t - yhat_base
        if X is not None:
            X_ctx = X[start:t, :]
            x_next = X[t : t + 1, :]  # next-time covariate (assumed known)
            # Standardize X within the context window to stabilize coefficients
            mu = X_ctx.mean(axis=0, keepdims=True)
            sd = X_ctx.std(axis=0, keepdims=True)
            sd = np.where(sd < 1e-6, 1.0, sd)
            Xn = (X_ctx - mu) / sd
            xn = (x_next - mu) / sd
            # Winsorize to curb outliers
            Xn = np.clip(Xn, -5.0, 5.0)
            xn = np.clip(xn, -5.0, 5.0)
            # Build residuals from precomputed base predictions over a limited window
            i0 = max(start + 1, t - ols_window)
            idx = np.arange(i0, t)
            valid = np.isfinite(base_preds_all[idx])
            idx = idx[valid]
            if idx.size > 0:
                # Residual at time i uses predictor X at the SAME time i
                r_targets = (y[idx] - base_preds_all[idx]).astype(np.float32)
                X_ols = Xn[(idx - start)]  # align: X at time i predicts y_i residual
            else:
                r_targets = np.array([], dtype=np.float32)
                X_ols = np.empty((0, Xn.shape[1]), dtype=np.float32)
            if len(X_ols) >= Xn.shape[1] + 2:
                X_mat = np.vstack(X_ols)
                y_vec = np.asarray(r_targets)
                # OLS with ridge epsilon for numerical stability
                lam = 1e-2
                beta = np.linalg.solve(X_mat.T @ X_mat + lam * np.eye(X_mat.shape[1]), X_mat.T @ y_vec)
                # Ensure 1D for stable scalar conversion
                delta = float(np.ravel(xn) @ beta)
                # Clip adjustment by residual variability in the window (robustness)
                r_std = float(np.std(y_vec)) if y_vec.size > 1 else 0.0
                clip_val = max(0.05, 3.0 * r_std)  # at least 0.05 level units
                if np.isfinite(clip_val):
                    delta = float(np.clip(delta, -clip_val, clip_val))
                yhat_adj = yhat_base + delta
            else:
                yhat_adj = yhat_base
        else:
            yhat_adj = yhat_base

        preds_base.append(yhat_base)
        preds.append(float(yhat_adj))
        stamps.append(d[date_col].iloc[t])

    out = pd.DataFrame({"timestamp": stamps, "model": preds, "model_base": preds_base, "actual": y[1:1+len(stamps)]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward forecasting on EUR/NOK with exogenous covariates adjustment")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", default="dataset_eksogene.csv", help="Path to CSV with DATE, EUR_NOK and covariates")
    ap.add_argument("--context_len", type=int, default=512)
    ap.add_argument("--covariates", type=str, default="Q,d_pi,dI_t", help="Comma-separated column names for covariates")
    ap.add_argument("--out", default="runs/wf_cov_eurnok")
    ap.add_argument("--anchor_to_last", type=str, default="true")
    # New options for strict quarter-ahead forecasting
    ap.add_argument("--to_quarter_end", action="store_true", help="Produce one forecast per quarter from origin (day before quarter start)")
    ap.add_argument("--cov_lag", type=int, default=0, help="Lag (days) applied to covariates; values beyond origin are frozen")
    ap.add_argument(
        "--cov_strategy",
        type=str,
        default="carryforward",
        choices=["carryforward", "zero"],
        help="How to handle covariates beyond origin in quarter-ahead mode",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Robust CSV read (infer separator)
    try:
        df = pd.read_csv(args.csv)
    except Exception:
        df = pd.read_csv(args.csv, sep=None, engine="python")
    # If we ended up with a single combined header (e.g., 'DATE;EUR_NOK;Q;d_pi;dI_t'), re-read with ';'
    if len(df.columns) == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(args.csv, sep=";")
    if isinstance(args.covariates, str) and args.covariates.strip().lower() in ("", "none", "null", "0"):
        cov_cols: List[str] = []
    else:
        cov_cols = [c.strip() for c in str(args.covariates).split(",") if c.strip()]

    model, dcfg, device = build_model_from_ckpt(args.ckpt)

    res = walk_forward_with_covariates(
        df,
        model,
        dcfg,
        date_col="DATE",
        price_col="EUR_NOK",
        covariate_cols=cov_cols,
        context_len=int(args.context_len),
        anchor_to_last=(args.anchor_to_last.lower() in ("1", "true", "yes")),
        ols_window=128,
    )

    out_csv = os.path.join(args.out, "walk_forward_cov.csv")
    res.to_csv(out_csv, index=False)

    # Plot actual vs predicted (adjusted)
    import matplotlib.pyplot as plt
    ts = pd.to_datetime(res["timestamp"], utc=True)
    plt.figure(figsize=(10, 5))
    plt.plot(ts, res["actual"], label="Actual (EUR/NOK)", color="black")
    plt.plot(ts, res["model"], label=" ", color="tab:blue", linestyle="--")
    plt.title("Walk-Forward with Covariates (EUR/NOK)")
    plt.ylabel("Level (EUR/NOK)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_img = os.path.join(args.out, "walk_forward_cov.jpeg")
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    try:
        plt.show()
    except Exception:
        pass
    print(f"Saved covariate walk-forward CSV and plot under {args.out}")

    # Optional: strict quarter-ahead from single origin per quarter
    if args.to_quarter_end:
        _quarter_ahead_from_origin(
            df=df,
            model=model,
            dcfg=dcfg,
            date_col="DATE",
            price_col="EUR_NOK",
            covariate_cols=cov_cols,
            context_len=int(args.context_len),
            anchor_to_last=(args.anchor_to_last.lower() in ("1", "true", "yes")),
            ols_window=128,
            cov_lag=int(args.cov_lag),
            cov_strategy=str(args.cov_strategy),
            out_dir=args.out,
        )


def _quarter_ahead_from_origin(
    *,
    df: pd.DataFrame,
    model: DecoderOnlyTSModel,
    dcfg: dict,
    date_col: str,
    price_col: str,
    covariate_cols: List[str],
    context_len: int,
    anchor_to_last: bool,
    ols_window: int,
    cov_lag: int,
    cov_strategy: str,
    out_dir: str,
) -> None:
    import matplotlib.pyplot as plt
    from fx_timesfm.evaluation import diebold_mariano

    d = _normalize_columns(df)
    date_col = _match_col(d, date_col, ["date", "Timestamp", "timestamp"])  # type: ignore[arg-type]
    price_col = _match_col(d, price_col, ["EUR/NOK", "eurnok", "price"])  # type: ignore[arg-type]
    d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, utc=True, errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    ts = d[date_col]
    y = d[price_col].astype(np.float32).to_numpy()

    Xdf = None
    if covariate_cols:
        cols_ok: List[str] = []
        for c in covariate_cols:
            try:
                cols_ok.append(_match_col(d, c, [c.replace("_", "").replace("/", "")]))
            except KeyError:
                # skip missing covariates silently
                continue
        if cols_ok:
            Xdf = d[cols_ok].astype(np.float32)

    # Precompute base one-step predictions for entire history (safe for indices <= origin)
    p = int(dcfg.get("input_patch_len", 32))
    device = next(model.parameters()).device
    base_hist = _compute_base_walk_forward(y, model, p, context_len, anchor_to_last=False)

    dQ = ts.dt.to_period("Q")
    quarters = dQ.unique().tolist()

    out_rows = []

    for q in quarters:
        # Quarter start and end
        q_mask = dQ == q
        if not np.any(q_mask):
            continue
        q_idx = np.where(q_mask.to_numpy())[0]
        q_start_idx = int(q_idx.min())
        q_end_idx = int(q_idx.max())
        # Origin is last observation strictly before quarter start
        origin_idx = q_start_idx - 1
        if origin_idx < 0:
            continue
        # Need enough context
        if origin_idx + 1 < p:
            continue

        # Build OLS fit on residuals up to origin
        i0 = max(1, origin_idx - ols_window + 1)
        idx = np.arange(i0, origin_idx + 1)
        valid = np.isfinite(base_hist[idx])
        idx = idx[valid]

        beta = None
        mu = None
        sd = None
        r_std = 0.0

        if Xdf is not None and len(idx) > 0:
            # Apply covariate lag
            Xlag = Xdf.shift(cov_lag)
            Xwin = Xlag.iloc[idx]
            # Standardize within window
            mu = Xwin.mean(axis=0)
            sd = Xwin.std(axis=0).replace(0.0, 1.0)
            Xn = (Xwin - mu) / sd
            Xn = Xn.clip(-5.0, 5.0)
            r_targets = (y[idx] - base_hist[idx]).astype(np.float32)
            if Xn.shape[0] >= Xn.shape[1] + 2:
                X_mat = Xn.to_numpy()
                y_vec = np.asarray(r_targets)
                lam = 1e-2
                beta = np.linalg.solve(X_mat.T @ X_mat + lam * np.eye(X_mat.shape[1]), X_mat.T @ y_vec)
                r_std = float(np.std(y_vec)) if y_vec.size > 1 else 0.0
        # Prepare covariate value to use beyond origin
        x_origin = None
        if Xdf is not None:
            Xlag_all = Xdf.shift(cov_lag)
            if origin_idx < len(Xlag_all):
                x_origin = Xlag_all.iloc[origin_idx:origin_idx + 1]
                if mu is not None and sd is not None:
                    x_origin = ((x_origin - mu) / sd).clip(-5.0, 5.0)

        # Recursive forecast from origin+1 to quarter end
        y_hist = y[: origin_idx + 1].astype(np.float32)
        last_actual = y_hist[-1]
        pred_qe = np.nan
        for t in range(origin_idx + 1, q_end_idx + 1):
            start = max(0, len(y_hist) - context_len)
            y_ctx = y_hist[start:]
            if len(y_ctx) < p:
                break
            patches = torch.from_numpy(patchify_edgepad(y_ctx, p)).unsqueeze(0).to(device)
            mean_s, std_s = scalar_stats(patches)
            patches_n = (patches - mean_s) / std_s
            with torch.no_grad():
                out = model(patches_n, attn_pad_mask=torch.zeros((1, patches.size(1)), dtype=torch.bool, device=device))
                yhat_n = out["pred"][:, -1, 0].item()
            yhat_base = float(yhat_n * std_s + mean_s)
            # Anchor only at first step to ensure continuity
            if t == origin_idx + 1 and anchor_to_last:
                yhat_base = float(last_actual)

            # Covariate adjustment using frozen coefficients and strategy
            if beta is not None and x_origin is not None and mu is not None and sd is not None:
                # Value available at future dates
                Xlag_all = Xdf.shift(cov_lag)
                x_t = Xlag_all.iloc[t: t + 1]
                if x_t.isna().values.any():
                    # Use strategy
                    if cov_strategy == "carryforward" and x_origin is not None:
                        xn = x_origin
                    else:
                        xn = pd.DataFrame(np.zeros_like(x_origin.to_numpy()), columns=x_origin.columns)
                else:
                    xn = ((x_t - mu) / sd).clip(-5.0, 5.0)
                delta = float(np.ravel(xn.to_numpy()) @ beta)
                clip_val = max(0.05, 3.0 * r_std)
                delta = float(np.clip(delta, -clip_val, clip_val))
                yhat = yhat_base + delta
            else:
                yhat = yhat_base

            y_hist = np.append(y_hist, [yhat]).astype(np.float32)
            if t == q_end_idx:
                pred_qe = float(yhat)

        if np.isfinite(pred_qe):
            ts_qe = ts.iloc[q_end_idx]
            actual_qe = float(y[q_end_idx]) if np.isfinite(y[q_end_idx]) else np.nan
            rw = float(y[origin_idx])  # Random-walk from origin
            out_rows.append({
                "timestamp": ts_qe,
                "model": pred_qe,
                "actual": actual_qe,
                "rw": rw,
            })

    if not out_rows:
        print("No quarter-ahead outputs produced (insufficient context or data).")
        return

    out_df = pd.DataFrame(out_rows).sort_values("timestamp").reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "quarterly_origin.csv")
    out_df.to_csv(out_csv, index=False)

    # Metrics
    msk = out_df["actual"].notna() & out_df["model"].notna() & out_df["rw"].notna()
    use = out_df[msk]
    y_true = use["actual"].to_numpy(dtype=float)
    y_hat = use["model"].to_numpy(dtype=float)
    y_rw = use["rw"].to_numpy(dtype=float)
    obs = int(len(y_true))
    mae = float(np.mean(np.abs(y_hat - y_true))) if obs else float("nan")
    rmse = float(np.sqrt(np.mean((y_hat - y_true) ** 2))) if obs else float("nan")
    dm_p = diebold_mariano(np.abs(y_hat - y_true) - np.abs(y_rw - y_true), h=1) if obs else float("nan")
    with open(os.path.join(out_dir, "quarterly_origin_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Observations: {obs}\n")
        f.write(f"MAE_model: {mae}\n")
        f.write(f"RMSE_model: {rmse}\n")
        f.write(f"DM_p (two-sided, abs loss): {dm_p}\n")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(use["timestamp"], use["actual"], label="Actual (EUR/NOK)", color="black")
    plt.plot(use["timestamp"], use["model"], label=" ", color="tab:blue", linestyle="--")
    plt.title("Quarter-Ahead (Origin) – Actual vs Predicted (EUR/NOK)")
    plt.ylabel("Level (EUR/NOK)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_img = os.path.join(out_dir, "quarterly_actual_vs_pred_origin.jpeg")
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    try:
        plt.show()
    except Exception:
        pass
    print(f"Saved quarter-ahead (origin) CSV/metrics/plot under {out_dir}")

if __name__ == "__main__":
    main()
