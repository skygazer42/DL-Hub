from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _spatial_kernel(radius: int, sigma: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    r = int(radius)
    if r < 0:
        raise ValueError("radius must be >= 0")
    s = float(sigma)
    if s <= 0.0:
        raise ValueError("sigma must be > 0")

    if r == 0:
        return torch.ones((1,), device=device, dtype=dtype)

    coords = torch.arange(-r, r + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    dist2 = xx * xx + yy * yy
    w = torch.exp(-0.5 * dist2 / (s * s))
    w = w.reshape(-1)
    return w / w.sum().clamp_min(1e-12)


class BilateralFilter(nn.Module):
    """Bilateral filter baseline (torch-only, toy-first).

    Applies edge-preserving smoothing by combining:
    - spatial weighting (distance in pixel grid)
    - range weighting (difference in intensity/color)

    This implementation is intended for small images (32-128px) and CPU use.
    """

    def __init__(
        self,
        *,
        radius: int = 3,
        spatial_sigma: float = 2.0,
        range_sigma: float = 0.1,
        iterations: int = 1,
        padding: str = "reflect",
        clamp: bool = True,
    ) -> None:
        super().__init__()
        r = int(radius)
        if r < 0:
            raise ValueError("radius must be >= 0")
        if float(range_sigma) <= 0.0:
            raise ValueError("range_sigma must be > 0")
        if float(spatial_sigma) <= 0.0:
            raise ValueError("spatial_sigma must be > 0")
        it = int(iterations)
        if it <= 0:
            raise ValueError("iterations must be > 0")

        self.radius = r
        self.spatial_sigma = float(spatial_sigma)
        self.range_sigma = float(range_sigma)
        self.iterations = it
        self.padding = str(padding)
        self.clamp = bool(clamp)

        # Register a buffer for spatial weights (created lazily per device/dtype).
        self.register_buffer("_spatial_w", torch.empty(0), persistent=False)

    def _get_spatial_w(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._spatial_w.numel() == 0 or self._spatial_w.device != device or self._spatial_w.dtype != dtype:
            w = _spatial_kernel(self.radius, self.spatial_sigma, device=device, dtype=dtype)
            self._spatial_w = w
        return self._spatial_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        if self.radius == 0:
            return x.clamp(0.0, 1.0) if self.clamp else x

        b, c, h, w = x.shape
        r = int(self.radius)
        k = 2 * r + 1
        center_idx = (k * k) // 2

        spatial_w = self._get_spatial_w(device=x.device, dtype=x.dtype).view(1, 1, k * k, 1)
        inv_2sig2 = 0.5 / (float(self.range_sigma) * float(self.range_sigma))

        y = x
        for _ in range(int(self.iterations)):
            y_pad = F.pad(y, (r, r, r, r), mode=self.padding)
            patches = F.unfold(y_pad, kernel_size=k)  # (B, C*k*k, H*W)
            patches = patches.view(b, c, k * k, h * w)
            center = patches[:, :, center_idx, :].unsqueeze(2)  # (B, C, 1, HW)
            diff2 = (patches - center).pow(2).sum(dim=1, keepdim=True)  # (B,1,K2,HW)
            range_w = torch.exp(-diff2 * float(inv_2sig2))
            w_all = spatial_w * range_w
            w_sum = w_all.sum(dim=2, keepdim=True).clamp_min(1e-12)
            y = (w_all * patches).sum(dim=2, keepdim=True) / w_sum
            y = y.squeeze(2).view(b, c, h, w)

        return y.clamp(0.0, 1.0) if self.clamp else y


_VARIANTS: dict[str, dict] = {
    "bilateral_fast": {"radius": 2, "spatial_sigma": 1.6, "iters": 1},
    "bilateral_quality": {"radius": 3, "spatial_sigma": 2.0, "iters": 1},
    "bilateral_strong": {"radius": 3, "spatial_sigma": 2.0, "iters": 2},
}


def build_bilateral_filter_denoiser(
    *,
    in_channels: int,  # unused (kept for consistent signatures)
    sigma: float = 0.1,
    variant: str = "bilateral_quality",
) -> nn.Module:
    _ = int(in_channels)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown BilateralFilter variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    # Heuristic mapping: use sigma as range sigma.
    range_sigma = max(1e-4, 1.25 * float(sigma))
    return BilateralFilter(
        radius=int(spec["radius"]),
        spatial_sigma=float(spec["spatial_sigma"]),
        range_sigma=float(range_sigma),
        iterations=int(spec["iters"]),
        padding="reflect",
        clamp=True,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.12).clamp(0.0, 1.0)
    m = build_bilateral_filter_denoiser(in_channels=1, sigma=0.12, variant="bilateral_fast")
    y = m(noisy)
    print("bilateral_fast", tuple(y.shape), float((y - x).pow(2).mean().item()))

