from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import torch
import pandas as pd
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm

from .data import DataConfig, make_dataloaders
from .masking import sample_r_mask_first_patch, build_causal_attention_mask
from .model import DecoderOnlyTSModel
from .normalization import compute_first_patch_stats, apply_first_patch_normalization, invert_first_patch_normalization
from .loss import combined_loss
from .metrics import mae
from .utils import load_config, seed_everything, ensure_dir, config_hash, save_checkpoint, build_cosine_scheduler
from .checkpoints import save_ckpt
from .model_card import ModelCard
from .evaluation import evaluate_model_vs_rw


PRESETS = {
    "small": {"d_model": 512, "n_layers": 8, "n_heads": 8},       # ~17M
    "medium": {"d_model": 896, "n_layers": 12, "n_heads": 14},    # ~70M approx
    "large": {"d_model": 1280, "n_layers": 20, "n_heads": 16},    # ~200M target
}


def cfg_to_dataconfig(cfg: Dict) -> DataConfig:
    d = cfg["data"]
    return DataConfig(
        train_paths=d["train_paths"],
        val_paths=d["val_paths"],
        test_paths=d.get("test_paths", []),
        symbols=d.get("symbols", []),
        timestamp_col=d.get("timestamp_col", "timestamp"),
        symbol_col=d.get("symbol_col", "symbol"),
        price_col=d.get("price_col", "price"),
        freq=d.get("freq", "1H"),
        resample=d.get("resample", True),
        fill_method=d.get("fill_method", "ffill"),
        context_len=d.get("context_len", 512),
        input_patch_len=d.get("input_patch_len", 32),
        output_patch_len=d.get("output_patch_len", 128),
        horizon=d.get("horizon", 256),
        max_context_patches=d.get("max_context_patches", 16),
        batch_size=d.get("batch_size", 8),
        num_workers=d.get("num_workers", 2),
    )


def build_model(cfg: Dict) -> DecoderOnlyTSModel:
    m = cfg["model"]
    preset = m.get("preset", "small")
    d_model = m.get("d_model", PRESETS[preset]["d_model"])
    n_layers = m.get("n_layers", PRESETS[preset]["n_layers"])
    n_heads = m.get("n_heads", PRESETS[preset]["n_heads"])
    dropout = m.get("dropout", 0.2)
    quantiles = bool(m.get("quantiles", False))
    quantile_levels = m.get("quantile_levels", [0.1, 0.5, 0.9])

    data = cfg["data"]
    model = DecoderOnlyTSModel(
        input_patch_len=data["input_patch_len"],
        output_patch_len=data["output_patch_len"],
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        quantiles=quantiles,
        quantile_levels=quantile_levels,
    )
    return model


def train_one_epoch(
    model: DecoderOnlyTSModel,
    loader,
    optimizer,
    scheduler,
    device,
    cfg: Dict,
    step_offset: int,
) -> Tuple[int, float]:
    model.train()
    log_interval = int(cfg["trainer"]["log_interval"])
    total_loss = 0.0
    steps = step_offset
    p = int(cfg["data"]["input_patch_len"])  # 32
    h = int(cfg["data"]["output_patch_len"])  # 128
    for it, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        patches = batch["patches"].to(device)  # [B, T, p]
        patch_pad_mask = batch["patch_pad_mask"].to(device)  # [B, T, p]
        attn_pad_mask = batch["attn_pad_mask"].to(device)  # [B, T]

        # normalization using first patch
        stats = compute_first_patch_stats(patches, patch_pad_mask)
        patches_norm = apply_first_patch_normalization(patches, stats)

        # masking first r
        obs_mask = sample_r_mask_first_patch(patches_norm, patch_pad_mask)  # [B, T, p]
        patches_masked = patches_norm * (1.0 - obs_mask)

        # build causal mask for T tokens
        T = patches_masked.size(1)
        causal_mask = build_causal_attention_mask(T, device)

        out = model(patches_masked, attn_pad_mask=attn_pad_mask, causal_mask=causal_mask)
        pred = out["pred"]  # [B, T, h]

        # Build training targets: for each token t, target is next h steps from raw series.
        # We can approximate by taking the next predicted patch equal to the next token's data (rolling nature).
        # Here, we supervise with the last token predicting zeros (no target); shift patches by 1 along token dim and pad 0.
        target = torch.zeros_like(pred)
        # Map next token's first h values by taking from patches_norm (denorm back to original scale for loss?)
        # We'll compute loss in normalized space; derive target by concatenating current+future raw points.
        # Simplify: target for token t is the next h points in original scale; we don't have those in patches; so use masked input for self-supervision:
        # To keep unit tests simple, we treat target as zeros here is not acceptable; instead we use the original patches to build pseudo targets:
        B, T, _ = patches.shape
        # Assemble per-token horizon from the underlying sequence constructed by concatenating tokens along patch axis
        seq = patches.reshape(B, T * p)
        for t in range(T):
            start = (t + 1) * p
            end = start + h
            tgt = torch.zeros((B, h), device=device, dtype=patches.dtype)
            if start < seq.size(1):
                tgt_slice = seq[:, start:end]
                tgt[:, : tgt_slice.size(1)] = tgt_slice
            target[:, t, :] = tgt

        # Normalize targets with same stats
        target_norm = (target - stats.mean) / stats.std

        # Loss mask: if horizon extends beyond sequence, those timesteps are masked out
        horizon_valid = torch.zeros_like(target_norm)
        for t in range(T):
            start = (t + 1) * p
            end = start + h
            valid_len = max(0, min(seq.size(1) - start, h))
            if valid_len > 0:
                horizon_valid[:, t, :valid_len] = 1.0
        loss_mask = 1.0 - horizon_valid  # 1 means ignore

        loss_dict = combined_loss(pred, target_norm, loss_mask, out.get("quantiles"), cfg["model"].get("quantile_levels"))
        loss = loss_dict["total"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=float(cfg["trainer"]["grad_clip_norm"]))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        steps += 1
        if steps % log_interval == 0:
            tqdm.write(f"step {steps} loss {loss.item():.4f}")
    return steps, total_loss / max(1, len(loader))


@torch.no_grad()
def validate(model: DecoderOnlyTSModel, loader, device, cfg: Dict) -> float:
    model.eval()
    p = int(cfg["data"]["input_patch_len"])  # 32
    h = int(cfg["data"]["output_patch_len"])  # 128
    maes: List[float] = []
    for batch in tqdm(loader, desc="val", leave=False):
        patches = batch["patches"].to(device)
        patch_pad_mask = batch["patch_pad_mask"].to(device)
        attn_pad_mask = batch["attn_pad_mask"].to(device)

        stats = compute_first_patch_stats(patches, patch_pad_mask)
        patches_norm = (patches - stats.mean) / stats.std

        T = patches_norm.size(1)
        causal_mask = build_causal_attention_mask(T, device)
        out = model(patches_norm, attn_pad_mask=attn_pad_mask, causal_mask=causal_mask)
        pred = out["pred"]  # [B, T, h]
        # target construction as in training
        B = patches.size(0)
        seq = patches.reshape(B, T * p)
        target = torch.zeros_like(pred)
        horizon_valid = torch.zeros_like(pred)
        for t in range(T):
            start = (t + 1) * p
            end = start + h
            if start < seq.size(1):
                tgt_slice = seq[:, start:end]
                target[:, t, : tgt_slice.size(1)] = tgt_slice
                horizon_valid[:, t, : tgt_slice.size(1)] = 1.0
        pred_denorm = pred * stats.std + stats.mean
        # compute MAE on valid positions only
        w = horizon_valid
        if w.sum() > 0:
            maes.append(((pred_denorm - target).abs() * w).sum().item() / w.sum().item())
    return float(sum(maes) / max(1, len(maes)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="fx_timesfm/config/default.yaml")
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.save_dir:
        cfg["save_dir"] = args.save_dir
    seed_everything(int(cfg.get("seed", 42)))
    # Prefer CUDA when available unless explicitly overridden
    if "device" in cfg and cfg["device"] in ("cpu", "cuda"):
        device = torch.device(cfg["device"])  # honor explicit device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg_to_dataconfig(cfg)
    train_loader, val_loader = make_dataloaders(data_cfg)

    model = build_model(cfg).to(device)
    total_steps = len(train_loader) * int(cfg["trainer"]["epochs"])
    optimizer = AdamW(model.parameters(), lr=float(cfg["trainer"]["lr"]), weight_decay=float(cfg["trainer"]["weight_decay"]), betas=tuple(cfg["trainer"]["betas"]))
    scheduler = None
    if cfg["trainer"].get("cosine_decay", True):
        scheduler = build_cosine_scheduler(optimizer, max_steps=total_steps, warmup_steps=int(cfg["trainer"]["warmup_steps"]))

    ckpt_dir = cfg.get("save_dir", cfg["trainer"].get("ckpt_dir", "checkpoints"))
    ensure_dir(ckpt_dir)
    best_gap = -float("inf")
    patience = int(cfg["trainer"].get("patience", cfg["trainer"].get("early_stop_patience", 5)))
    wait = 0
    global_step = 0
    for epoch in range(int(cfg["trainer"]["epochs"])):
        global_step, train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, cfg, global_step)
        # Evaluate vs RW monthly/quarterly (macro)
        # Build a minimal DataFrame from the validation loader to reuse evaluation helper
        # Here we reconstruct from underlying datasets for simplicity in this minimal impl
        from .data import load_price_frames
        dcfg = cfg["data"]
        # Robustly find validation paths
        if isinstance(dcfg.get("val_paths"), list) and len(dcfg["val_paths"]) > 0:
            val_paths = dcfg["val_paths"]
        else:
            # backward compat
            fallback = dcfg.get("val_csv", "data/val.csv")
            val_paths = [fallback]
        val_df = load_price_frames(val_paths, "timestamp", "symbol", "price")
        val_df["timestamp"] = pd.to_datetime(val_df["timestamp"], utc=True)
        eval_res = evaluate_model_vs_rw(model, val_df, p=int(dcfg["input_patch_len"]), h=int(dcfg["output_patch_len"]), context_len=int(dcfg["context_len"]), device=device)
        gap = eval_res.get("monthly", {}).get("mae_gap_vs_rw", float("nan"))
        # Fallback: if monthly gap not computable (e.g., too-short validation span), fallback to negative val MAE surrogate
        if gap != gap:  # NaN check
            val_mae = validate(model, val_loader, device, cfg)
            gap = -val_mae
            print(f"epoch {epoch} train_loss {train_loss:.4f} mae_gap_vs_rw_monthly NaN (fallback to -val_mae={-gap:.4f})")
        else:
            print(f"epoch {epoch} train_loss {train_loss:.4f} mae_gap_vs_rw_monthly {gap:.4f}")
        if gap > best_gap:
            best_gap = gap
            wait = 0
            # Save with ModelCard
            card = ModelCard.create(
                framework="pytorch",
                d_model=model.d_model,
                n_layers=len(model.blocks),
                n_heads=model.blocks[0].attn.num_heads if hasattr(model.blocks[0].attn, 'num_heads') else 0,
                input_patch_len=model.p,
                output_patch_len=model.h,
                context_len_max=cfg["data"]["context_len"],
                normalization="first_patch_mean_std",
                train_freq=cfg["data"].get("freq", "D"),
                train_date_range=(dcfg.get("min_timestamp", "2000-01-01"), "present"),
                train_symbols=[],
                metrics_snapshot={"mae_gap_vs_rw_monthly": float(gap)},
            )
            path = os.path.join(ckpt_dir, "best.ckpt")
            save_ckpt(path, model.state_dict(), optimizer.state_dict(), cfg, card)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping (no improvement in mae_gap_vs_rw_monthly).")
                break


if __name__ == "__main__":
    main()
