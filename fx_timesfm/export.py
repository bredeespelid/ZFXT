from __future__ import annotations

import os
from typing import Any, Dict

import torch

from .model import DecoderOnlyTSModel
from .utils import ensure_dir


def save_model(ckpt_path: str, model: DecoderOnlyTSModel, extra: Dict[str, Any] | None = None) -> None:
    ensure_dir(os.path.dirname(ckpt_path) or ".")
    state = {
        "model_state": model.state_dict(),
        "model_cfg": {
            "input_patch_len": model.p,
            "output_patch_len": model.h,
            "d_model": model.d_model,
        },
    }
    if extra:
        state.update(extra)
    torch.save(state, ckpt_path)


def load_model(ckpt_path: str, d_model: int, n_layers: int, n_heads: int, dropout: float, quantiles: bool, quantile_levels: list[float]) -> DecoderOnlyTSModel:
    payload = torch.load(ckpt_path, map_location="cpu")
    cfg = payload.get("model_cfg", {})
    model = DecoderOnlyTSModel(
        input_patch_len=cfg.get("input_patch_len", 32),
        output_patch_len=cfg.get("output_patch_len", 128),
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        quantiles=quantiles,
        quantile_levels=quantile_levels,
    )
    model.load_state_dict(payload["model_state"])  # type: ignore[arg-type]
    return model


def to_torchscript(model: DecoderOnlyTSModel, example_input: torch.Tensor, path: str) -> None:
    model.eval()
    traced = torch.jit.trace(model, (example_input, torch.zeros((example_input.size(0), example_input.size(1)), dtype=torch.bool)))
    ensure_dir(os.path.dirname(path) or ".")
    traced.save(path)
