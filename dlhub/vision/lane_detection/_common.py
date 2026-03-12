import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                kernel_size=3,
                stride=int(stride),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(int(out_channels)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        c = int(channels)
        self.conv1 = ConvBNAct(c, c)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.dropout = nn.Dropout2d(float(dropout))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.dropout(y)
        return self.act(x + y)


class TinyLaneEncoder(nn.Module):
    """A tiny multi-scale encoder that returns stride-4 and stride-8 features."""

    def __init__(
        self,
        *,
        in_channels: int,
        stem_channels: int,
        hidden_channels: int,
        depth: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        stem = int(stem_channels)
        hidden = int(hidden_channels)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), stem, stride=2),
            ResidualBlock(stem, dropout=float(dropout)),
        )
        self.low = nn.Sequential(
            ConvBNAct(stem, hidden, stride=2),
            *[ResidualBlock(hidden, dropout=float(dropout)) for _ in range(d)],
        )
        self.high = nn.Sequential(
            ConvBNAct(hidden, hidden, stride=2),
            *[ResidualBlock(hidden, dropout=float(dropout)) for _ in range(d)],
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        low = self.low(x)
        high = self.high(low)
        return low, high


class SegmentationDecoder(nn.Module):
    def __init__(
        self,
        *,
        low_channels: int,
        high_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.low_proj = ConvBNAct(int(low_channels), int(out_channels))
        self.high_proj = ConvBNAct(int(high_channels), int(out_channels))
        self.fuse = nn.Sequential(
            ConvBNAct(int(out_channels) * 2, int(out_channels)),
            ResidualBlock(int(out_channels), dropout=float(dropout)),
        )

    def forward(
        self,
        low: torch.Tensor,
        high: torch.Tensor,
        *,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        low = self.low_proj(low)
        high = self.high_proj(high)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fuse(torch.cat([low, high], dim=1))
        return F.interpolate(fused, size=output_size, mode="bilinear", align_corners=False)


class SpatialMessagePassing(nn.Module):
    """A lightweight approximation of SCNN-style directional propagation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.row_conv = nn.Conv2d(c, c, kernel_size=(1, 5), padding=(0, 2), groups=c, bias=False)
        self.col_conv = nn.Conv2d(c, c, kernel_size=(5, 1), padding=(2, 0), groups=c, bias=False)
        self.mix = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.row_conv(x)
        y = y + self.col_conv(y)
        y = self.mix(y)
        y = self.bn(y)
        return self.act(x + y)


class GlobalContextHead(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int, *, dropout: float) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.net = nn.Sequential(
            nn.Linear(int(in_channels), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(x).flatten(1)
        return self.net(pooled)


def scaled_channels(channels: int, width_mult: float, *, min_ch: int = 16) -> int:
    return scale_channels(int(channels), float(width_mult), min_ch=min_ch, divisor=8)


def choose_attention_heads(embed_dim: int, *, target_head_dim: int = 16) -> int:
    dim = int(embed_dim)
    if dim <= 0:
        raise ValueError("embed_dim must be > 0")

    heads = max(1, dim // int(target_head_dim))
    while heads > 1 and dim % heads != 0:
        heads -= 1
    return heads


def choose_even_attention_heads(embed_dim: int, *, target_head_dim: int = 16) -> int:
    dim = int(embed_dim)
    if dim <= 0:
        raise ValueError("embed_dim must be > 0")

    heads = choose_attention_heads(dim, target_head_dim=int(target_head_dim))
    if heads % 2 == 0:
        return heads

    for candidate in range(heads + 1, dim + 1):
        if candidate % 2 == 0 and dim % candidate == 0:
            return candidate

    for candidate in range(heads - 1, 1, -1):
        if candidate % 2 == 0 and dim % candidate == 0:
            return candidate

    return heads
