import numpy as np
from fx_timesfm.data import patchify


def test_patchify_basic():
    x = np.arange(10, dtype=np.float32)
    patches, pad_mask = patchify(x, patch_len=4)
    assert patches.shape == (3, 4)
    assert pad_mask.sum() == 2  # padded two zeros
