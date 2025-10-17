import numpy as np
import torch
from src.model.decoder_only import TimesFMDecoder
from src.model.layers import TransformerConfig
from src.data.patch_dataset import PatchDataset

def test_model_forward():
    config = TransformerConfig(d_model=64, n_heads=4, n_layers=2, ffn_dim=128, dropout=0.0, context_len=64)
    model = TimesFMDecoder(in_dim=10, config=config)
    x = torch.randn(2, 64, 10)
    y, h = model(x)
    assert y.shape == (2, 64, 10)
    assert h.shape == (2, 64, 64)


def test_patch_dataset():
    arr = np.random.randn(200, 10).astype(np.float32)
    ds = PatchDataset(arr, context_len=64, patch_len=8, stride=4)
    item = ds[0]
    assert item['input'].shape == (64, 10)
    assert item['target'].shape == (64, 10)
    assert item['time_mask'].shape == (64,)
