from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import torch
from torch import Tensor, nn

from .masking import build_causal_attention_mask


class ResidualBlock(nn.Module):
    """Simple MLP residual block that operates on patch dimension.

    Input: [B, T, P]
    """

    def __init__(self, p: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj_in = nn.Linear(p, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, p)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x: [B, T, P]
        h = self.proj_in(x)
        h2 = self.mlp(self.ln1(h))
        h = h + h2
        out = self.proj_out(self.ln2(h))
        return h, out  # return token embeddings h and reconstructed patches out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        # x: [B, T, D]
        h = self.ln1(x)
        y, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + y
        y = self.ff(self.ln2(x))
        x = x + y
        return x


class DecoderOnlyTSModel(nn.Module):
    def __init__(
        self,
        input_patch_len: int,
        output_patch_len: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dropout: float = 0.1,
        quantiles: bool = False,
        quantile_levels: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.p = input_patch_len
        self.h = output_patch_len
        self.d_model = d_model
        self.quantiles = quantiles
        self.quantile_levels = quantile_levels or [0.1, 0.5, 0.9]

        self.input_block = ResidualBlock(self.p, d_model, dropout)
        self.pos_enc = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        # Output projection: token embedding -> output patch of size h
        self.out_block = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.h),
        )
        if self.quantiles:
            self.q_head = nn.Linear(self.h, self.h * len(self.quantile_levels))
        else:
            self.q_head = None

    def forward(
        self,
        patches: Tensor,  # [B, T_p, p]
        attn_pad_mask: Optional[Tensor],  # [B, T_p] with True for padded tokens
        causal_mask: Optional[Tensor] = None,  # [T_p, T_p] with -inf for masked positions
    ) -> Dict[str, Tensor]:
        B, T, P = patches.shape
        token_emb, _ = self.input_block(patches)
        token_emb = self.pos_enc(token_emb)

        x = token_emb
        for blk in self.blocks:
            x = blk(x, attn_mask=causal_mask, key_padding_mask=attn_pad_mask)
        x = self.ln_f(x)

        # Predict next patch for each token position
        out_patches = self.out_block(x)  # [B, T, h]
        # quantiles per step
        out_quantiles = None
        if self.q_head is not None:
            q = self.q_head(out_patches)  # [B, T, h*Q]
            Q = len(self.quantile_levels)
            out_quantiles = q.view(B, T, self.h, Q)

        return {"pred": out_patches, "quantiles": out_quantiles}

    @torch.no_grad()
    def generate(self, context: Tensor, steps: int, device: Optional[torch.device] = None) -> Tensor:
        """Autoregressive decoding by appending h-sized outputs.

        Args:
            context: [1, T_p, p] context patches
            steps: total horizon length to generate (in points), multiples of h recommended
        Returns:
            preds: [steps] tensor on cpu
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        context = context.to(device)
        h = self.h
        out: List[Tensor] = []
        cur = context.clone()
        generated = 0
        while generated < steps:
            T = cur.size(1)
            causal = build_causal_attention_mask(T, device)
            attn_pad = torch.zeros((1, T), dtype=torch.bool, device=device)
            o = self.forward(cur, attn_pad_mask=attn_pad, causal_mask=causal)["pred"]  # [1, T, h]
            next_patch = o[:, -1, :]  # [1, h]
            out.append(next_patch.view(-1))
            generated += next_patch.numel()
            # Append as a new token: need to map predictions back to patch space P to continue
            # We simply take first p points of the generated horizon to form next input token
            next_token = torch.zeros((1, 1, self.p), device=device)
            take = min(self.p, h)
            next_token[:, 0, :take] = next_patch[:, :take]
            cur = torch.cat([cur, next_token], dim=1)
        pred_series = torch.cat(out, dim=0)[:steps].detach().cpu()
        return pred_series
