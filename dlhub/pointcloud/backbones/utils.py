from __future__ import annotations

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


def _c(ch: int, width_mult: float, *, min_ch: int = 8, divisor: int = 8) -> int:
    v = max(int(min_ch), int(round(int(ch) * float(width_mult))))
    return _make_divisible(v, int(divisor))


class ConvBNAct1d(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        act: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        act_name = str(act).lower().strip()
        if act_name in {"relu"}:
            act_layer: nn.Module = nn.ReLU(inplace=True)
        elif act_name in {"gelu"}:
            act_layer = nn.GELU()
        elif act_name in {"leaky", "leakyrelu"}:
            act_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unknown activation: {act!r}")

        layers: list[nn.Module] = [
            nn.Conv1d(int(in_ch), int(out_ch), kernel_size=1, bias=False),
            nn.BatchNorm1d(int(out_ch)),
            act_layer,
        ]
        if float(dropout) > 0:
            layers.append(nn.Dropout(p=float(dropout)))
        super().__init__(*layers)


class ConvBNAct2d(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        act: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        act_name = str(act).lower().strip()
        if act_name in {"relu"}:
            act_layer: nn.Module = nn.ReLU(inplace=True)
        elif act_name in {"gelu"}:
            act_layer = nn.GELU()
        elif act_name in {"leaky", "leakyrelu"}:
            act_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unknown activation: {act!r}")

        layers: list[nn.Module] = [
            nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(out_ch)),
            act_layer,
        ]
        if float(dropout) > 0:
            layers.append(nn.Dropout2d(p=float(dropout)))
        super().__init__(*layers)


def global_max_pool(x: torch.Tensor) -> torch.Tensor:
    """Pool (B, C, N) -> (B, C)."""

    if x.ndim != 3:
        raise ValueError(f"Expected (B, C, N), got {tuple(x.shape)}")
    return torch.max(x, dim=-1).values


__all__ = [
    "ConvBNAct1d",
    "ConvBNAct2d",
    "_c",
    "global_max_pool",
]

