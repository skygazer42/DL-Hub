from __future__ import annotations


import torch
import torch.nn.functional as F
from torch import nn


def check_low_res_image(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
    if int(x.shape[1]) <= 0:
        raise ValueError(f"Expected C > 0, got {int(x.shape[1])}")
    if int(x.shape[-2]) < 4 or int(x.shape[-1]) < 4:
        raise ValueError(f"Expected spatial dims >= 4, got {(int(x.shape[-2]), int(x.shape[-1]))}")
    return x


def validate_upscale_factor(upscale_factor: int) -> int:
    scale = int(upscale_factor)
    if scale < 2:
        raise ValueError(f"upscale_factor must be >= 2, got {scale}")
    if scale != 2:
        raise ValueError(f"Only upscale_factor=2 is supported in v1, got {scale}")
    return scale


def _default_variants(prefix: str) -> dict[str, dict[str, int]]:
    name = str(prefix).strip().lower()
    if not name:
        raise ValueError("prefix must be a non-empty string")
    return {
        f"{name}_tiny": {"width": 16, "depth": 2},
        f"{name}_small": {"width": 24, "depth": 4},
        f"{name}_base": {"width": 32, "depth": 6},
    }


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, res_scale: float = 1.0) -> None:
        super().__init__()
        c = int(channels)
        if c <= 0:
            raise ValueError("channels must be > 0")
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.res_scale = float(res_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.conv1(x))
        y = self.conv2(y)
        return x + y * self.res_scale


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, *, reduction: int = 8) -> None:
        super().__init__()
        c = int(channels)
        r = max(1, int(reduction))
        hidden = max(4, c // r)
        if c <= 0:
            raise ValueError("channels must be > 0")
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Conv2d(c, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = F.gelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class PixelShuffleUpsampler(nn.Module):
    def __init__(self, channels: int, *, upscale_factor: int = 2) -> None:
        super().__init__()
        c = int(channels)
        scale = validate_upscale_factor(upscale_factor)
        if c <= 0:
            raise ValueError("channels must be > 0")
        self.proj = nn.Conv2d(c, c * scale * scale, kernel_size=3, padding=1, bias=True)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.proj(x))


def bicubic_upsample(x: torch.Tensor, *, upscale_factor: int = 2) -> torch.Tensor:
    scale = validate_upscale_factor(upscale_factor)
    return F.interpolate(x, scale_factor=float(scale), mode="bicubic", align_corners=False)


def compute_psnr(
    pred: torch.Tensor, target: torch.Tensor, *, data_range: float = 1.0
) -> torch.Tensor:
    pred = pred.to(torch.float32)
    target = target.to(torch.float32)
    mse = F.mse_loss(pred, target)
    eps = torch.tensor(1e-8, dtype=pred.dtype, device=pred.device)
    return 10.0 * torch.log10(
        torch.tensor(float(data_range * data_range), dtype=pred.dtype, device=pred.device)
        / torch.clamp(mse, min=float(eps))
    )


def num_parameters(module: nn.Module) -> int:
    return int(sum(int(p.numel()) for p in module.parameters()))


__all__ = [
    "ChannelAttention",
    "PixelShuffleUpsampler",
    "ResidualBlock",
    "_default_variants",
    "bicubic_upsample",
    "check_low_res_image",
    "compute_psnr",
    "num_parameters",
    "validate_upscale_factor",
]
