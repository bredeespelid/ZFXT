import torch
from fx_timesfm.masking import sample_r_mask_first_patch


def test_sample_r_mask_first_patch_shapes():
    B, T, p = 2, 3, 4
    patches = torch.randn(B, T, p)
    pad_mask = torch.zeros(B, T, p)
    obs_mask = sample_r_mask_first_patch(patches, pad_mask)
    assert obs_mask.shape == (B, T, p)
    # Only first patch may have masked prefix besides pad
    assert (obs_mask[:, 1:, :] == 0).all()
