from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

from fx_timesfm.checkpoints import load_ckpt, save_ckpt
from fx_timesfm.model import DecoderOnlyTSModel
from fx_timesfm.data import patchify


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save", default=None)
    args = p.parse_args()

    payload, card = load_ckpt(args.ckpt)
    cfg = payload["cfg"]
    mcfg = cfg["model"]
    dcfg = cfg["data"]
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

    # Freeze transformer blocks
    for p_ in model.blocks.parameters():
        p_.requires_grad = False
    for p_ in model.pos_enc.parameters():
        p_.requires_grad = False
    for p_ in model.ln_f.parameters():
        p_.requires_grad = False

    # Train only input and output blocks
    params = list(model.input_block.parameters()) + list(model.out_block.parameters())
    opt = AdamW(params, lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def load_series(path: str):
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    train = load_series(args.train_csv)
    val = load_series(args.val_csv)

    # Minimal loop: iterate symbols, take last 512 points, patchify, and train for alignment
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for sym, g in train.groupby("symbol"):
            y = g["price"].values.astype("float32")
            if len(y) < 64:
                continue
            patches, pad = patchify(y[-512:], 32)
            x = torch.from_numpy(patches).unsqueeze(0).to(device)
            attn_pad = torch.zeros((1, x.size(1)), dtype=torch.bool, device=device)
            out = model(x, attn_pad, None)
            loss = out["pred"].pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"epoch {epoch} adapt loss {np.mean(losses) if losses else float('nan')}")

    save_path = args.save or os.path.splitext(args.ckpt)[0] + ".adapted.ckpt"
    save_ckpt(save_path, model.state_dict(), None, cfg, card)
    print(f"Saved adapted checkpoint to {save_path}")


if __name__ == "__main__":
    main()
