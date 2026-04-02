from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct


def check_nchw(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
    return x


def logits_to_parsing(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError(f"logits must have shape (B, K, H, W), got {tuple(logits.shape)}")
    return logits.argmax(dim=1)


def make_coord_grid(
    batch: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    yy = torch.linspace(-1.0, 1.0, steps=int(height), device=device, dtype=dtype)
    xx = torch.linspace(-1.0, 1.0, steps=int(width), device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
    return coords.expand(int(batch), -1, -1, -1)


def make_tanh_warp_grid(
    batch: int,
    height: int,
    width: int,
    *,
    focus: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = make_coord_grid(
        batch,
        height,
        width,
        device=device,
        dtype=dtype,
    )
    scale = max(0.5, float(focus))
    warp = torch.tanh(coords * scale) / torch.tanh(torch.tensor(scale, device=device, dtype=dtype))
    return warp.permute(0, 2, 3, 1).contiguous()


class TinyFaceEncoder(nn.Module):
    """Small multi-scale encoder for face parsing toy models."""

    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        c = int(width)
        d = max(1, int(depth))
        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), c, kernel_size=3, stride=2, act="relu"),
            ConvBNAct(c, c, kernel_size=3, stride=1, act="relu"),
        )
        self.stage2 = self._stage(c, c * 2, depth=d, dropout=float(dropout))
        self.stage3 = self._stage(c * 2, c * 4, depth=d, dropout=float(dropout))
        self.out_channels = (c, c * 2, c * 4)

    @staticmethod
    def _stage(in_ch: int, out_ch: int, *, depth: int, dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = [ConvBNAct(int(in_ch), int(out_ch), kernel_size=3, stride=2, act="relu")]
        for _ in range(max(1, int(depth)) - 1):
            layers.append(ConvBNAct(int(out_ch), int(out_ch), kernel_size=3, stride=1, act="relu"))
            if float(dropout) > 0:
                layers.append(nn.Dropout2d(float(dropout)))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = check_nchw(x)
        c1 = self.stem(x)  # /2
        c2 = self.stage2(c1)  # /4
        c3 = self.stage3(c2)  # /8
        return c1, c2, c3


class ParsingHead(nn.Module):
    def __init__(self, *, in_channels: int, hidden_channels: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(int(in_channels), int(hidden_channels), kernel_size=3, stride=1, act="relu"),
            nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Conv2d(int(hidden_channels), int(num_classes), kernel_size=1, bias=True),
        )

    def forward(self, feat: torch.Tensor, *, out_hw: tuple[int, int]) -> torch.Tensor:
        logits = self.net(feat)
        return F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
