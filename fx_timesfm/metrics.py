from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def mae(y_hat: Tensor, y: Tensor) -> Tensor:
    return (y_hat - y).abs().mean()


def smape(y_hat: Tensor, y: Tensor, eps: float = 1e-6) -> Tensor:
    denom = (y_hat.abs() + y.abs()).clamp_min(eps)
    return (2.0 * (y_hat - y).abs() / denom).mean()


def msmape(y_hat: Tensor, y: Tensor, eps: float = 1e-3) -> Tensor:
    denom = y.abs().clamp_min(eps)
    return ((y_hat - y).abs() / denom).mean()


def mase(y_hat: Tensor, y: Tensor, y_insample: Tensor, m: int = 1, eps: float = 1e-6) -> Tensor:
    # Mean absolute scaled error with seasonal period m
    diff = (y_insample[m:] - y_insample[:-m]).abs().mean().clamp_min(eps)
    return (y_hat - y).abs().mean() / diff


METRIC_FUNCS = {
    "mae": mae,
    "smape": smape,
    "msmape": msmape,
    "mase": mase,
}
