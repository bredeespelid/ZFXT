from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config with recursive inheritance via 'inherits' key.

    If a config includes `inherits: parent` or `inherits: parent.yaml`, the parent
    is loaded first (recursively) and then overridden by the child.
    """
    cfg = _load_yaml(path)
    base_dir = os.path.dirname(path)

    if "inherits" in cfg and cfg["inherits"]:
        parent = cfg.pop("inherits")
        parent_path = os.path.join(base_dir, f"{parent}.yaml") if not str(parent).endswith(".yaml") else os.path.join(base_dir, str(parent))
        base_cfg = load_config(parent_path)  # recursive resolution
        cfg = _merge_dicts(base_cfg, cfg)
    return cfg


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def config_hash(cfg: Dict[str, Any]) -> str:
    s = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    torch.save(state, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def build_cosine_scheduler(optimizer: torch.optim.Optimizer, max_steps: int, warmup_steps: int = 0, min_lr: float = 0.0):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress)) * (1.0 - min_lr) + min_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
