import torch
from fx_timesfm.normalization import compute_first_patch_stats, apply_first_patch_normalization, invert_first_patch_normalization


def test_norm_roundtrip():
    B, T, p = 3, 5, 7
    patches = torch.randn(B, T, p)
    pad_mask = torch.zeros(B, T, p)
    stats = compute_first_patch_stats(patches, pad_mask)
    y = apply_first_patch_normalization(patches, stats)
    z = invert_first_patch_normalization(y, stats)
    assert torch.allclose(patches, z, atol=1e-5)
