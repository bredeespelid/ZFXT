"""
fx_timesfm: Minimal decoder-only foundation model for FX time-series forecasting.

This package provides:
- Data utilities and datasets
- Normalization with first-patch scaling
- Decoder-only Transformer model with patch tokens
- Masking utilities for variable context exposure
- Training and inference CLIs
- Metrics and export helpers
"""

__all__ = [
    "data",
    "normalization",
    "model",
    "masking",
    "loss",
    "train",
    "infer",
    "utils",
    "metrics",
    "export",
]
