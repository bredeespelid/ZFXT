from __future__ import annotations
import os
import math
import argparse
import yaml
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..model.decoder_only import TimesFMDecoder
from ..model.layers import TransformerConfig
from ..data.preprocess import load_dataset, normalize_and_split
from ..data.patch_dataset import create_dataloader
from ..utils.io import ensure_dir, save_checkpoint


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    # pred/target: (B, T, C), mask: (B, T)
    diff = (pred - target) ** 2
    maskf = mask.float().unsqueeze(-1)
    loss = (diff * maskf).sum() / (maskf.sum() * pred.shape[-1] + 1e-8)
    return loss


def load_config():
    cfg_path = os.environ.get('CFG', 'configs/default.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train():
    cfg = load_config()
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    ensure_dir(cfg['paths']['checkpoints_dir'])
    ensure_dir(cfg['paths']['outputs_dir'])

    # data
    df = load_dataset(cfg['paths']['data'])
    (train_arr, _), (val_arr, _), mean, std = normalize_and_split(df, cfg['paths']['norm_stats'], cfg['train']['val_split'], cfg['seed'])

    train_loader = create_dataloader(
        train_arr,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        context_len=cfg['model']['context_len'],
        patch_len=cfg['model']['patch_len'],
        stride=cfg['model']['stride'],
        mask_ratio_min=cfg['model']['mask_ratio_min'],
        mask_ratio_max=cfg['model']['mask_ratio_max'],
    )
    val_loader = create_dataloader(
        val_arr,
        batch_size=cfg['train']['batch_size'],
        shuffle=False,
        num_workers=cfg['train']['num_workers'],
        context_len=cfg['model']['context_len'],
        patch_len=cfg['model']['patch_len'],
        stride=cfg['model']['stride'],
        mask_ratio_min=cfg['model']['mask_ratio_min'],
        mask_ratio_max=cfg['model']['mask_ratio_max'],
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_cfg = TransformerConfig(
        d_model=cfg['model']['d_model'],
        n_heads=cfg['model']['n_heads'],
        n_layers=cfg['model']['n_layers'],
        ffn_dim=cfg['model']['ffn_dim'],
        dropout=cfg['model']['dropout'],
        context_len=cfg['model']['context_len'],
    )
    model = TimesFMDecoder(in_dim=cfg['model']['in_dim'], config=model_cfg).to(device)

    opt = AdamW(model.parameters(), lr=cfg['train']['lr'], betas=tuple(cfg['train']['betas']), weight_decay=cfg['train']['weight_decay'])
    scaler = GradScaler(enabled=cfg['train']['amp'] and device=='cuda')

    # cosine with T_max as total steps per epoch * epochs; but here we update per step
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg['train']['epochs']
    scheduler = CosineAnnealingLR(opt, T_max=total_steps)

    writer = SummaryWriter(log_dir='runs/foundation_timesfm')

    best_val = float('inf')
    patience = cfg['train']['early_stop_patience']
    bad_epochs = 0

    global_step = 0
    for epoch in range(cfg['train']['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        running = 0.0
        for batch in pbar:
            x = batch['input'].to(device)  # (B, T, C)
            t = batch['target'].to(device)
            m = batch['time_mask'].to(device)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=cfg['train']['amp'] and device=='cuda'):
                y, _ = model(x)
                loss = masked_mse(y, t, m)
            scaler.scale(loss).backward()
            if cfg['train']['grad_clip']:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            running += loss.item()
            if global_step % cfg['train']['log_interval'] == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['input'].to(device)
                t = batch['target'].to(device)
                m = batch['time_mask'].to(device)
                y, _ = model(x)
                l = masked_mse(y, t, m)
                val_losses.append(l.item())
        val_loss = float(np.mean(val_losses))
        writer.add_scalar('val/loss', val_loss, epoch)
        print(f"Epoch {epoch} val_loss={val_loss:.4f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            bad_epochs = 0
            save_checkpoint(
                os.path.join(cfg['paths']['checkpoints_dir'], 'foundation_decoder.pt'),
                model.state_dict(),
                optimizer_state=opt.state_dict(),
                extra={
                    'config': cfg,
                }
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    writer.close()


if __name__ == '__main__':
    train()
