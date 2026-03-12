from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.to(torch.float32)
        denom = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x_fp32 / denom).to(dtype=x.dtype) * self.weight


def make_attention_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones_like(input_ids, dtype=torch.float32)
    return attention_mask.to(dtype=torch.float32, device=input_ids.device)


def causal_mask(seq_len: int, *, device: torch.device) -> torch.Tensor:
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).tril()
    return mask.view(1, 1, seq_len, seq_len)


def causal_mask_with_offset(
    query_len: int,
    key_len: int,
    *,
    device: torch.device,
    query_offset: int = 0,
) -> torch.Tensor:
    q_positions = torch.arange(
        int(query_offset),
        int(query_offset) + int(query_len),
        device=device,
    )[:, None]
    k_positions = torch.arange(int(key_len), device=device)[None, :]
    mask = k_positions <= q_positions
    return mask.view(1, 1, int(query_len), int(key_len))


def expand_key_padding_mask(mask: torch.Tensor, *, batch_size: int, seq_len: int) -> torch.Tensor:
    if mask.shape != (batch_size, seq_len):
        raise ValueError(
            f"attention_mask must be {(batch_size, seq_len)}, got {tuple(mask.shape)}"
        )
    return mask.to(torch.bool).view(batch_size, 1, 1, seq_len)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _build_rotary_cache(
    seq_len: int,
    rotary_dim: int,
    *,
    device: torch.device,
    base: float,
    position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be even")
    positions = torch.arange(
        int(position_offset),
        int(position_offset) + int(seq_len),
        device=device,
        dtype=torch.float32,
    )
    inv_freq = 1.0 / (
        float(base) ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim)
    )
    freqs = torch.einsum("t,d->td", positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().view(1, 1, seq_len, rotary_dim), emb.sin().view(1, 1, seq_len, rotary_dim)


def apply_rotary_embeddings(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    rotary_dim: int,
    base: float = 10000.0,
    position_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rotary_dim <= 0:
        return q, k
    if rotary_dim > q.shape[-1] or rotary_dim > k.shape[-1]:
        raise ValueError("rotary_dim cannot exceed head_dim")
    if q.shape[-2] != k.shape[-2]:
        raise ValueError("q and k must have the same sequence length for rotary application")
    cos, sin = _build_rotary_cache(
        q.shape[-2],
        rotary_dim,
        device=q.device,
        base=float(base),
        position_offset=int(position_offset),
    )

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


def build_alibi_slopes(num_heads: int) -> torch.Tensor:
    n = int(num_heads)
    if n <= 0:
        raise ValueError("num_heads must be > 0")

    def power_of_two_slopes(m: int) -> list[float]:
        start = 2.0 ** (-(2.0 ** (-(math.log2(m) - 3))))
        ratio = start
        return [start * (ratio**i) for i in range(m)]

    if math.log2(n).is_integer():
        slopes = power_of_two_slopes(n)
    else:
        closest = 2 ** int(math.floor(math.log2(n)))
        slopes = power_of_two_slopes(closest)
        extra = power_of_two_slopes(2 * closest)
        slopes.extend(extra[0::2][: n - closest])
    return torch.tensor(slopes, dtype=torch.float32)


def build_alibi_bias(
    num_heads: int,
    seq_len: int,
    *,
    device: torch.device,
    key_len: int | None = None,
    query_offset: int = 0,
) -> torch.Tensor:
    slopes = build_alibi_slopes(num_heads).to(device=device)
    target_len = int(seq_len if key_len is None else key_len)
    q_positions = torch.arange(
        int(query_offset),
        int(query_offset) + int(seq_len),
        device=device,
        dtype=torch.float32,
    )[:, None]
    k_positions = torch.arange(target_len, device=device, dtype=torch.float32)[None, :]
    rel = k_positions - q_positions
    bias = -rel.abs().view(1, 1, int(seq_len), target_len)
    return bias * slopes.view(1, num_heads, 1, 1)


class SwiGLUMLP(nn.Module):
    activation_name = "swiglu"

    def __init__(self, dim: int, intermediate_size: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(int(dim), int(intermediate_size), bias=False)
        self.up_proj = nn.Linear(int(dim), int(intermediate_size), bias=False)
        self.down_proj = nn.Linear(int(intermediate_size), int(dim), bias=False)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class GELUMLP(nn.Module):
    activation_name = "gelu"

    def __init__(self, dim: int, intermediate_size: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.dense_h_to_4h = nn.Linear(int(dim), int(intermediate_size))
        self.dense_4h_to_h = nn.Linear(int(intermediate_size), int(dim))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.dense_h_to_4h(x))
        x = self.dropout(x)
        return self.dense_4h_to_h(x)
