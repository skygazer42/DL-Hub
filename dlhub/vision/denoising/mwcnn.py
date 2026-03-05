from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ._utils import pad_to_multiple, unpad


class HaarDWT2D(nn.Module):
    """2D Haar DWT implemented as grouped strided conv (orthonormal, toy-first)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        if c <= 0:
            raise ValueError("channels must be > 0")

        # (4, 1, 2, 2)
        k = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],  # LL
                [[1.0, -1.0], [1.0, -1.0]],  # LH
                [[1.0, 1.0], [-1.0, -1.0]],  # HL
                [[1.0, -1.0], [-1.0, 1.0]],  # HH
            ],
            dtype=torch.float32,
        )
        k = k / 2.0
        weight = k.view(4, 1, 2, 2).repeat(c, 1, 1, 1)  # (4C, 1, 2, 2)
        self.register_buffer("weight", weight)
        self.channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected NCHW, got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected C={self.channels}, got C={int(x.shape[1])}")
        if (x.shape[-2] % 2) != 0 or (x.shape[-1] % 2) != 0:
            raise ValueError("H and W must be even for HaarDWT2D")
        w = self.weight.to(dtype=x.dtype)
        return F.conv2d(x, w, stride=2, padding=0, groups=self.channels)


class HaarIDWT2D(nn.Module):
    """2D Haar inverse DWT implemented as grouped conv_transpose2d."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        if c <= 0:
            raise ValueError("channels must be > 0")

        k = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],  # LL
                [[1.0, -1.0], [1.0, -1.0]],  # LH
                [[1.0, 1.0], [-1.0, -1.0]],  # HL
                [[1.0, -1.0], [-1.0, 1.0]],  # HH
            ],
            dtype=torch.float32,
        )
        k = k / 2.0
        weight = k.view(4, 1, 2, 2).repeat(c, 1, 1, 1)  # (4C, 1, 2, 2)
        self.register_buffer("weight", weight)
        self.channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected NCHW, got {tuple(x.shape)}")
        if x.shape[1] != self.channels * 4:
            raise ValueError(f"Expected C={self.channels*4}, got C={int(x.shape[1])}")
        w = self.weight.to(dtype=x.dtype)
        return F.conv_transpose2d(x, w, stride=2, padding=0, groups=self.channels)


class _ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        return F.relu(x + y, inplace=True)


class MWCNN(nn.Module):
    """MWCNN (Multi-level Wavelet CNN) toy-first denoiser.

    Uses a Haar wavelet down/up transform to get multi-scale features without explicit pooling.
    Predicts a residual/noise map and returns `x - residual`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        features: int = 24,
        num_blocks: int = 3,
        mid_blocks: int = 3,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        nb = int(num_blocks)
        mb = int(mid_blocks)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f < 8:
            raise ValueError("features must be >= 8")
        if nb <= 0 or mb <= 0:
            raise ValueError("num_blocks/mid_blocks must be > 0")

        self.intro = nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True)
        self.body0 = nn.Sequential(*[_ResBlock(f) for _ in range(nb)])

        self.dwt = HaarDWT2D(f)
        self.mid = nn.Sequential(*[_ResBlock(f * 4) for _ in range(mb)])
        self.idwt = HaarIDWT2D(f)

        self.fuse = nn.Conv2d(f, f, kernel_size=3, padding=1, bias=True)
        self.outro = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        inp = x
        x_pad, pad_hw = pad_to_multiple(x, 2, mode="reflect")

        f0 = F.relu(self.intro(x_pad), inplace=True)
        f0 = self.body0(f0)

        y = self.dwt(f0)
        y = self.mid(y)
        y = self.idwt(y)

        y = F.relu(self.fuse(y + f0), inplace=True)
        residual = self.outro(y)
        out = x_pad - residual
        return unpad(out, pad_hw)


_VARIANTS: dict[str, dict] = {
    "mwcnn_tiny": {"features": 16, "num_blocks": 2, "mid_blocks": 2},
    "mwcnn_small": {"features": 24, "num_blocks": 3, "mid_blocks": 3},
    "mwcnn_base": {"features": 32, "num_blocks": 4, "mid_blocks": 4},
}


def build_mwcnn_denoiser(*, in_channels: int, variant: str = "mwcnn_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MWCNN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return MWCNN(
        in_channels=int(in_channels),
        features=int(spec["features"]),
        num_blocks=int(spec["num_blocks"]),
        mid_blocks=int(spec["mid_blocks"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_mwcnn_denoiser(in_channels=1, variant="mwcnn_tiny")
    y = m(noisy)
    print("mwcnn_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

