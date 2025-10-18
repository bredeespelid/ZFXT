from __future__ import annotations

import argparse
from typing import Dict, List

import torch
from tqdm import tqdm

from .utils import load_config, seed_everything
from .train import cfg_to_dataconfig, build_model
from .data import make_dataloaders
from .masking import build_causal_attention_mask
from .normalization import compute_first_patch_stats
from .metrics import mae, msmape


@torch.no_grad()
def evaluate_last_window(cfg_path: str, split: str = "val") -> Dict[str, float]:
    cfg = load_config(cfg_path)
    seed_everything(int(cfg.get("seed", 42)))
    device = torch.device(cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")
    data_cfg = cfg_to_dataconfig(cfg)
    train_loader, val_loader = make_dataloaders(data_cfg)
    loader = val_loader if split != "train" else train_loader

    model = build_model(cfg).to(device)
    # try to load best checkpoint if available
    import os
    from .utils import load_checkpoint

    ckpt_path = os.path.join(cfg["trainer"]["ckpt_dir"], "model_best.pt")
    if os.path.exists(ckpt_path):
        payload = load_checkpoint(ckpt_path)
        model.load_state_dict(payload["model_state"])  # type: ignore[arg-type]
    model.eval()

    p = int(cfg["data"]["input_patch_len"])  # 32
    h = int(cfg["data"]["output_patch_len"])  # 128
    maes: List[float] = []
    msm: List[float] = []

    for batch in tqdm(loader, desc=f"bench-{split}"):
        patches = batch["patches"].to(device)
        patch_pad_mask = batch["patch_pad_mask"].to(device)
        attn_pad_mask = batch["attn_pad_mask"].to(device)

        stats = compute_first_patch_stats(patches, patch_pad_mask)
        patches_norm = (patches - stats.mean) / stats.std
        T = patches_norm.size(1)
        causal_mask = build_causal_attention_mask(T, device)
        out = model(patches_norm, attn_pad_mask=attn_pad_mask, causal_mask=causal_mask)
        pred = out["pred"]  # [B, T, h]

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

        # evaluate only last token per sample for last-window performance
        last_pred = pred_denorm[:, -1, :]
        last_tgt = target[:, -1, :]
        last_w = horizon_valid[:, -1, :]
        # flatten valid positions
        mask = last_w > 0
        if mask.sum() == 0:
            continue
        y_hat = last_pred[mask]
        y = last_tgt[mask]
        maes.append(float(mae(y_hat, y)))
        msm.append(float(msmape(y_hat, y)))

    return {"mae": sum(maes) / max(1, len(maes)), "msmape": sum(msm) / max(1, len(msm))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="fx_timesfm/config/default.yaml")
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()
    scores = evaluate_last_window(args.config, args.split)
    print(scores)


if __name__ == "__main__":
    main()
