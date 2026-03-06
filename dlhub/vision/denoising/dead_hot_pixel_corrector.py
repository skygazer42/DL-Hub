from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _median_filter(x: torch.Tensor, *, k: int, padding: str) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
    k = int(k)
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be odd and >= 1")
    if k == 1:
        return x

    p = k // 2
    b, c, h, w = x.shape
    x_pad = F.pad(x, (p, p, p, p), mode=str(padding))
    patches = F.unfold(x_pad, kernel_size=k)  # (B, C*k*k, H*W)
    patches = patches.view(b, c, k * k, h * w)
    y = patches.median(dim=2).values.view(b, c, h, w)
    return y


class DeadHotPixelCorrector(nn.Module):
    """Repair fixed dead/hot pixels by detecting isolated extremes and replacing with local median.

    This is designed to work well with `noise_type=dead_hot` in Lesson 10.
    """

    def __init__(
        self,
        *,
        kernel_size: int = 3,
        diff_threshold: float = 0.4,
        max_support: int = 2,
        iterations: int = 1,
        padding: str = "reflect",
        clamp: bool = True,
    ) -> None:
        super().__init__()
        k = int(kernel_size)
        if k < 1 or k % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 1")
        t = float(diff_threshold)
        if t <= 0.0:
            raise ValueError("diff_threshold must be > 0")
        ms = int(max_support)
        if ms < 1:
            raise ValueError("max_support must be >= 1")
        it = int(iterations)
        if it <= 0:
            raise ValueError("iterations must be > 0")

        self.kernel_size = k
        self.diff_threshold = t
        self.max_support = ms
        self.iterations = it
        self.padding = str(padding)
        self.clamp = bool(clamp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        y = x
        for _ in range(int(self.iterations)):
            med = _median_filter(y, k=int(self.kernel_size), padding=self.padding)
            diff = (y - med).abs()
            # Detect extreme pixels, but avoid clobbering legitimate edges that also contain many 0/1 values.
            eps = 1e-4
            hot = y > (1.0 - eps)
            dead = y < eps
            extreme = hot | dead

            # "Isolated" check: dead/hot pixels are usually sparse; strong edges are not.
            k = int(self.kernel_size)
            p = k // 2
            b, c, _, _ = y.shape
            ones = torch.ones((c, 1, k, k), device=y.device, dtype=y.dtype)
            hot_count = F.conv2d(F.pad(hot.to(y.dtype), (p, p, p, p), mode=self.padding), ones, groups=c)
            dead_count = F.conv2d(F.pad(dead.to(y.dtype), (p, p, p, p), mode=self.padding), ones, groups=c)
            isolated = torch.where(hot, hot_count, dead_count) <= float(self.max_support)

            mask = extreme & isolated & (diff > float(self.diff_threshold))
            y = torch.where(mask, med, y)

        return y.clamp(0.0, 1.0) if self.clamp else y


_VARIANTS: dict[str, dict] = {
    "dead_hot_tiny": {"k": 3, "iters": 1},
    "dead_hot_small": {"k": 3, "iters": 2},
    "dead_hot_base": {"k": 5, "iters": 2},
}


def build_dead_hot_pixel_corrector_denoiser(
    *,
    in_channels: int,  # unused (kept for consistent signatures)
    sigma: float = 0.1,
    variant: str = "dead_hot_small",
) -> nn.Module:
    _ = int(in_channels)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DeadHotPixelCorrector variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    # Use sigma to scale the defect detection strictness (larger sigma -> tolerate bigger deviations).
    diff_thr = max(0.25, min(0.8, 2.5 * float(sigma) + 0.15))
    return DeadHotPixelCorrector(
        kernel_size=int(spec["k"]),
        diff_threshold=float(diff_thr),
        max_support=2 if int(spec["k"]) <= 3 else 6,
        iterations=int(spec["iters"]),
        padding="reflect",
        clamp=True,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    clean = torch.zeros(1, 1, 32, 32)
    clean[:, :, 8:24, 8:24] = 1.0
    noisy = clean.clone()
    # Inject a few hot/dead pixels.
    noisy[:, :, 5, 5] = 1.0
    noisy[:, :, 10, 10] = 0.0
    noisy[:, :, 20, 12] = 0.0
    m = build_dead_hot_pixel_corrector_denoiser(in_channels=1, sigma=0.1, variant="dead_hot_tiny")
    out = m(noisy)
    print("dead_hot_tiny", tuple(out.shape), float((out - clean).abs().mean().item()))
