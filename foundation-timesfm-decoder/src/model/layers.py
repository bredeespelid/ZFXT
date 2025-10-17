from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ffn_dim: int = 1024
    dropout: float = 0.1
    context_len: int = 512


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, context_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # causal mask as buffer
        mask = torch.tril(torch.ones(context_len, context_len)).view(1, 1, context_len, context_len)
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config.d_model, config.n_heads, config.dropout, config.context_len)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config.d_model, config.ffn_dim, config.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, context_len: int, d_model: int):
        super().__init__()
        self.pos_emb = nn.Embedding(context_len, d_model)

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
        return x + self.pos_emb(pos).expand(B, T, C)
