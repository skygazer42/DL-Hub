from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MedianFilter(nn.Module):
    """Median filter baseline (torch-only, toy-first).

    This is a simple non-learnable denoiser that can be useful for impulse noise.
    """

    def __init__(self, *, kernel_size: int = 3, padding: str = "reflect", clamp: bool = True) -> None:
        super().__init__()
        k = int(kernel_size)
        if k < 1 or k % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 1")
        self.kernel_size = k
        self.padding = str(padding)
        self.clamp = bool(clamp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        b, c, h, w = x.shape
        k = int(self.kernel_size)
        p = k // 2

        if k == 1:
            return x.clamp(0.0, 1.0) if self.clamp else x

        x_pad = F.pad(x, (p, p, p, p), mode=self.padding)
        patches = F.unfold(x_pad, kernel_size=k)  # (B, C*k*k, H*W)
        patches = patches.reshape(b, c, k * k, h * w)
        y = patches.median(dim=2).values.reshape(b, c, h, w)
        return y.clamp(0.0, 1.0) if self.clamp else y


_VARIANTS: dict[str, dict] = {
    "median_tiny": {"k": 3},
    "median_small": {"k": 5},
    "median_base": {"k": 7},
}


def build_median_filter_denoiser(
    *,
    in_channels: int,  # unused (kept for consistent signatures)
    sigma: float = 0.1,  # unused (kept for consistent signatures)
    variant: str = "median_tiny",
) -> nn.Module:
    _ = int(in_channels)
    _ = float(sigma)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MedianFilter variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return MedianFilter(kernel_size=int(spec["k"]), padding="reflect", clamp=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 32, 32)
    noisy = (x + (torch.rand_like(x) < 0.05).to(x.dtype) * 0.8).clamp(0.0, 1.0)
    m = build_median_filter_denoiser(in_channels=1, variant="median_tiny")
    y = m(noisy)
    print("median_tiny", tuple(y.shape), float((y - x).abs().mean().item()))

