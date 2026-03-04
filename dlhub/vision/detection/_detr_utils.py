from __future__ import annotations

import math

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, num_layers: int = 2, act: str = "relu") -> None:
        super().__init__()
        d_in = int(in_dim)
        d_h = int(hidden_dim)
        d_out = int(out_dim)
        n = int(num_layers)
        if n <= 0:
            raise ValueError("num_layers must be > 0")

        act_name = str(act).lower().strip()
        if act_name == "relu":
            act_layer: nn.Module = nn.ReLU(inplace=True)
        elif act_name == "gelu":
            act_layer = nn.GELU()
        elif act_name in {"silu", "swish"}:
            act_layer = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {act!r}")

        layers: list[nn.Module] = []
        if n == 1:
            layers.append(nn.Linear(d_in, d_out))
        else:
            layers.append(nn.Linear(d_in, d_h))
            layers.append(act_layer)
            for _ in range(n - 2):
                layers.append(nn.Linear(d_h, d_h))
                layers.append(act_layer)
            layers.append(nn.Linear(d_h, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiheadAttention(nn.Module):
    """A tiny, dependency-free multi-head attention (supports self/cross attention).

    Shapes:
      - q: (B, Nq, D)
      - kv: (B, Nk, D)
    """

    def __init__(self, dim: int, num_heads: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        if d <= 0:
            raise ValueError("dim must be > 0")
        if h <= 0:
            raise ValueError("num_heads must be > 0")
        if d % h != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = d
        self.num_heads = h
        self.head_dim = d // h
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d, d, bias=True)
        self.k_proj = nn.Linear(d, d, bias=True)
        self.v_proj = nn.Linear(d, d, bias=True)
        self.out = nn.Linear(d, d, bias=True)
        self.attn_drop = nn.Dropout(p=float(dropout))
        self.proj_drop = nn.Dropout(p=float(dropout))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        b, nq, d = q.shape
        b2, nk, d2 = kv.shape
        if b2 != b:
            raise ValueError("Batch size mismatch between q and kv")
        if d != self.dim or d2 != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got q_dim={d}, kv_dim={d2}")

        q = self.q_proj(q).view(b, nq, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,Nq,Dh)
        k = self.k_proj(kv).view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,Nk,Dh)
        v = self.v_proj(kv).view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,Nk,Dh)

        attn = torch.matmul(q, k.transpose(-2, -1)) * float(self.scale)  # (B,H,Nq,Nk)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        y = torch.matmul(attn, v)  # (B,H,Nq,Dh)
        y = y.transpose(1, 2).contiguous().view(b, nq, d)  # (B,Nq,D)
        y = self.out(y)
        return self.proj_drop(y)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = nn.LayerNorm(d)
        self.attn = MultiheadAttention(d, int(num_heads), dropout=float(dropout))
        self.drop1 = nn.Dropout(p=float(dropout))
        self.norm2 = nn.LayerNorm(d)
        hidden = int(round(d * float(mlp_ratio)))
        self.mlp = MLP(d, hidden, d, num_layers=2, act="relu")
        self.drop2 = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.norm1(x), self.norm1(x)))
        x = x + self.drop2(self.mlp(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = nn.LayerNorm(d)
        self.self_attn = MultiheadAttention(d, int(num_heads), dropout=float(dropout))
        self.drop1 = nn.Dropout(p=float(dropout))

        self.norm2 = nn.LayerNorm(d)
        self.cross_attn = MultiheadAttention(d, int(num_heads), dropout=float(dropout))
        self.drop2 = nn.Dropout(p=float(dropout))

        self.norm3 = nn.LayerNorm(d)
        hidden = int(round(d * float(mlp_ratio)))
        self.mlp = MLP(d, hidden, d, num_layers=2, act="relu")
        self.drop3 = nn.Dropout(p=float(dropout))

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        tgt = tgt + self.drop1(self.self_attn(self.norm1(tgt), self.norm1(tgt)))
        tgt = tgt + self.drop2(self.cross_attn(self.norm2(tgt), memory))
        tgt = tgt + self.drop3(self.mlp(self.norm3(tgt)))
        return tgt


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d = int(dim)
        self.encoder = nn.ModuleList(
            [EncoderLayer(d, int(num_heads), mlp_ratio=float(mlp_ratio), dropout=float(dropout)) for _ in range(int(num_encoder_layers))]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d, int(num_heads), mlp_ratio=float(mlp_ratio), dropout=float(dropout)) for _ in range(int(num_decoder_layers))]
        )

    def forward(self, memory: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        # memory: (B, N, D); queries: (B, Q, D)
        for layer in self.encoder:
            memory = layer(memory)
        tgt = queries
        for layer in self.decoder:
            tgt = layer(tgt, memory)
        return tgt


def flatten_hw(feat: torch.Tensor) -> torch.Tensor:
    # (B, D, H, W) -> (B, HW, D)
    if feat.ndim != 4:
        raise ValueError(f"Expected NCHW tensor, got {tuple(feat.shape)}")
    return feat.flatten(2).transpose(1, 2).contiguous()


def sine_positional_encoding_1d(length: int, dim: int, *, device: torch.device) -> torch.Tensor:
    """Sinusoidal positional encoding (1D).

    Returned shape: (length, dim).
    """

    n = int(length)
    d = int(dim)
    if n <= 0:
        raise ValueError("length must be > 0")
    if d <= 0:
        raise ValueError("dim must be > 0")
    if d % 2 != 0:
        raise ValueError("dim must be even")

    pe = torch.zeros(n, d, device=device)
    pos = torch.arange(0, n, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

