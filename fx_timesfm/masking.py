from __future__ import annotations

import torch
from torch import Tensor


def sample_r_mask_first_patch(patches: Tensor, pad_mask: Tensor) -> Tensor:
    """Mask first r points in the first patch to expose variable context lengths.

    Args:
        patches: [B, T_p, p]
        pad_mask: [B, T_p, p] (1 for padded positions)
    Returns:
        patch_obs_mask: [B, T_p, p] with 1 where masked (either sampled r or pad), 0 where observed.
    """
    B, T, p = patches.shape
    device = patches.device
    # sample r in [0, p-1]
    r = torch.randint(low=0, high=p, size=(B,), device=device)
    obs_mask = pad_mask.clone()
    for b in range(B):
        if T > 0 and r[b] > 0:
            obs_mask[b, 0, : r[b]] = 1.0
    return obs_mask


def build_causal_attention_mask(T: int, device: torch.device) -> Tensor:
    """Build [T, T] boolean mask with True above diagonal for causal self-attention.

    Note: Using a boolean mask aligns the dtype with key_padding_mask (bool),
    avoiding PyTorch's deprecation warning about mismatched mask types.
    """
    # True indicates positions to mask (disallow attending to future tokens)
    mask = torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1)
    return mask


def merge_attention_padding(attn_pad_mask: Tensor) -> Tensor:
    """Convert [B, T] bool padding mask to additive mask [B, 1, 1, T] with -inf for pads."""
    B, T = attn_pad_mask.shape
    mask = attn_pad_mask.float() * -1e9
    return mask.view(B, 1, 1, T)
