import torch
from fx_timesfm.model import DecoderOnlyTSModel
from fx_timesfm.masking import build_causal_attention_mask


def test_forward_shapes():
    B, T, p, h = 2, 4, 8, 16
    model = DecoderOnlyTSModel(input_patch_len=p, output_patch_len=h, d_model=64, n_layers=2, n_heads=4)
    patches = torch.randn(B, T, p)
    causal = build_causal_attention_mask(T, device=patches.device)
    attn_pad = torch.zeros(B, T, dtype=torch.bool)
    out = model(patches, attn_pad, causal)
    assert out["pred"].shape == (B, T, h)
