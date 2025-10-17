import os
import json
import numpy as np
import torch


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(path: str, model_state: dict, optimizer_state: dict | None = None, extra: dict | None = None):
    ensure_dir(os.path.dirname(path))
    payload = {
        'model': model_state,
        'optimizer': optimizer_state,
        'extra': extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str | torch.device = 'cpu'):
    payload = torch.load(path, map_location=map_location)
    return payload.get('model'), payload.get('optimizer'), payload.get('extra', {})


def save_norm_stats(path: str, mean: np.ndarray, std: np.ndarray):
    ensure_dir(os.path.dirname(path))
    np.save(path, {'mean': mean.astype(np.float32), 'std': std.astype(np.float32)})


def load_norm_stats(path: str):
    obj = np.load(path, allow_pickle=True).item()
    return obj['mean'], obj['std']


def save_json(path: str, data: dict):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
