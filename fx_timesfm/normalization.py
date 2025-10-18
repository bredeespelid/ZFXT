from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor


@dataclass
class FirstPatchStats:
    mean: Tensor  # [B, 1, 1]
    std: Tensor   # [B, 1, 1]


def compute_first_patch_stats(patches: Tensor, patch_mask: Tensor) -> FirstPatchStats:
    """Compute mean/std from first input patch only (ignoring masked positions).

    Args:
        patches: [B, T_p, p]
        patch_mask: [B, T_p, p] with 1 where masked/padded, 0 where valid.
    Returns:
        FirstPatchStats with shape [B, 1, 1].
    """
    first_patch = patches[:, :1, :]  # [B, 1, p]
    first_mask = patch_mask[:, :1, :]  # [B, 1, p]
    valid = 1.0 - first_mask
    denom = torch.clamp(valid.sum(dim=(1, 2), keepdim=True), min=1.0)
    mean = (first_patch * valid).sum(dim=(1, 2), keepdim=True) / denom
    var = ((first_patch - mean) * valid).pow(2).sum(dim=(1, 2), keepdim=True) / denom
    std = torch.sqrt(var + 1e-6)
    return FirstPatchStats(mean=mean, std=std)


def apply_first_patch_normalization(patches: Tensor, stats: FirstPatchStats) -> Tensor:
    """Apply reversible instance normalization using provided stats.

    Args:
        patches: [B, T_p, p]
        stats: FirstPatchStats with [B,1,1]
    """
    return (patches - stats.mean) / stats.std


def invert_first_patch_normalization(patches: Tensor, stats: FirstPatchStats) -> Tensor:
    """Invert normalization.

    Args:
        patches: [B, T_p, p]
    """
    return patches * stats.std + stats.mean
