from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from .layers import TransformerConfig, DecoderBlock, PositionalEmbedding


class TimesFMDecoder(nn.Module):
    def __init__(self, in_dim: int = 206, config: TransformerConfig | None = None):
        super().__init__()
        if config is None:
            config = TransformerConfig()
        self.config = config
        self.input_proj = nn.Linear(in_dim, config.d_model)
        self.pos = PositionalEmbedding(config.context_len, config.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, in_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # x: (B, T, in_dim)
        x = self.input_proj(x)
        x = self.pos(x)
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        y = self.output_proj(x)
        return y, x  # return outputs and latent states

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
