from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor


def mse_loss(pred: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean squared error with optional mask.

    Args:
        pred: [...]
        target: [...]
        mask: same shape broadcastable; 1 means ignore, 0 means include.
    """
    diff = (pred - target) ** 2
    if mask is not None:
        w = 1.0 - mask
        num = (diff * w).sum()
        den = torch.clamp(w.sum(), min=1.0)
        return num / den
    return diff.mean()


def quantile_pinball_loss(pred_q: Tensor, target: Tensor, taus: List[float], mask: Tensor | None = None) -> Tensor:
    """Pinball loss for multiple quantiles.

    pred_q: [..., Q]
    target: [...]
    """
    Q = len(taus)
    target = target.unsqueeze(-1).expand_as(pred_q)
    diff = target - pred_q
    taus_t = torch.tensor(taus, device=pred_q.device, dtype=pred_q.dtype)
    loss = torch.maximum(taus_t * diff, (taus_t - 1.0) * diff)
    if mask is not None:
        w = (1.0 - mask).unsqueeze(-1)
        num = (loss * w).sum()
        den = torch.clamp(w.sum(), min=1.0)
        return num / den
    return loss.mean()


def combined_loss(
    pred_point: Tensor,
    target: Tensor,
    mask: Tensor | None,
    pred_quantiles: Tensor | None,
    taus: List[float] | None,
    alpha: float = 0.5,
) -> Dict[str, Tensor]:
    """Compute point MSE and optional quantile loss; return dict of components and total."""
    mse = mse_loss(pred_point, target, mask)
    out = {"mse": mse}
    if pred_quantiles is not None and taus is not None:
        ql = quantile_pinball_loss(pred_quantiles, target, taus, mask)
        total = alpha * mse + (1 - alpha) * ql
        out["quantile"] = ql
        out["total"] = total
    else:
        out["total"] = mse
    return out
