from __future__ import annotations

import math

import torch
from torch import nn


def _make_divisible(v: int, divisor: int = 8) -> int:
    d = int(divisor)
    if d <= 0:
        raise ValueError("divisor must be > 0")
    x = int(v)
    if x <= 0:
        return d
    return int((x + d - 1) // d * d)


def _d(dim: int, width_mult: float, *, min_dim: int = 32, divisor: int = 8) -> int:
    v = max(int(min_dim), int(round(int(dim) * float(width_mult))))
    return _make_divisible(v, int(divisor))


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool `(B, T, C)` into `(B, C)` using mask `(B, T)` in {0,1}."""

    if x.ndim != 3:
        raise ValueError(f"x must be (B, T, C), got {tuple(x.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be (B, T), got {tuple(mask.shape)}")
    if mask.dtype != torch.float32:
        mask = mask.to(torch.float32)

    w = mask.unsqueeze(-1)  # (B, T, 1)
    summed = (x * w).sum(dim=1)
    denom = w.sum(dim=1).clamp(min=1.0)
    return summed / denom


def masked_max_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Max-pool `(B, T, C)` into `(B, C)` with padding masked out."""

    if x.ndim != 3:
        raise ValueError(f"x must be (B, T, C), got {tuple(x.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be (B, T), got {tuple(mask.shape)}")

    key_mask = mask.to(torch.bool).unsqueeze(-1)  # (B, T, 1)
    x = x.masked_fill(~key_mask, float("-inf"))
    pooled = x.max(dim=1).values
    pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
    return pooled


def sequence_lengths(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be (B, T), got {tuple(attention_mask.shape)}")
    return attention_mask.to(torch.long).sum(dim=1).clamp(min=1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"RMSNorm expected last dim {self.dim}, got {x.shape[-1]}")
        x = x.to(torch.float32)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        y = x / rms
        return y * self.weight


def build_sinusoidal_positions(
    max_length: int, dim: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return (max_length, dim) sinusoidal position embeddings."""

    t = int(max_length)
    d = int(dim)
    if t <= 0:
        raise ValueError("max_length must be > 0")
    if d <= 0:
        raise ValueError("dim must be > 0")

    half = d // 2
    pos = torch.arange(t, device=device, dtype=torch.float32).unsqueeze(1)
    freq = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, half - 1))
    )
    angles = pos * freq.unsqueeze(0)
    pe = torch.zeros((t, d), device=device, dtype=torch.float32)
    pe[:, 0:half] = torch.sin(angles)
    pe[:, half : 2 * half] = torch.cos(angles)
    if d % 2 == 1:
        pe[:, -1] = 0.0
    return pe.to(dtype=dtype)


__all__ = [
    "RMSNorm",
    "_d",
    "_make_divisible",
    "build_sinusoidal_positions",
    "masked_max_pool",
    "masked_mean_pool",
    "sequence_lengths",
]

