from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _median_filter_h(x: torch.Tensor, *, k: int, padding: str) -> torch.Tensor:
    """Median filter along H for NCHW tensors (W can be 1)."""

    if x.ndim != 4:
        raise ValueError(f"Expected NCHW, got {tuple(x.shape)}")
    k = int(k)
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be odd and >= 1")
    if k == 1:
        return x

    p = k // 2
    x_pad = F.pad(x, (0, 0, p, p), mode=str(padding))
    patches = F.unfold(x_pad, kernel_size=(k, 1))  # (B, C*k, H)
    b, ck, l = patches.shape
    c = ck // k
    patches = patches.view(b, c, k, l)
    y = patches.median(dim=2).values.view(b, c, l, 1)
    return y


def _median_filter_w(x: torch.Tensor, *, k: int, padding: str) -> torch.Tensor:
    """Median filter along W for NCHW tensors (H can be 1)."""

    if x.ndim != 4:
        raise ValueError(f"Expected NCHW, got {tuple(x.shape)}")
    k = int(k)
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be odd and >= 1")
    if k == 1:
        return x

    p = k // 2
    x_pad = F.pad(x, (p, p, 0, 0), mode=str(padding))
    patches = F.unfold(x_pad, kernel_size=(1, k))  # (B, C*k, W)
    b, ck, l = patches.shape
    c = ck // k
    patches = patches.view(b, c, k, l)
    y = patches.median(dim=2).values.view(b, c, 1, l)
    return y


class LineDefectCorrector(nn.Module):
    """Repair stuck row/column lines by detecting flat lines that deviate from neighbors.

    Designed for `noise_type=line_defect` (fixed-pattern stuck rows/cols).
    """

    def __init__(
        self,
        *,
        window: int = 7,
        var_threshold: float = 1e-6,
        mean_threshold: float = 0.2,
        iterations: int = 1,
        padding: str = "reflect",
        clamp: bool = True,
    ) -> None:
        super().__init__()
        w = int(window)
        if w < 1 or w % 2 == 0:
            raise ValueError("window must be odd and >= 1")
        if float(var_threshold) <= 0.0:
            raise ValueError("var_threshold must be > 0")
        if float(mean_threshold) <= 0.0:
            raise ValueError("mean_threshold must be > 0")
        it = int(iterations)
        if it <= 0:
            raise ValueError("iterations must be > 0")
        self.window = w
        self.var_threshold = float(var_threshold)
        self.mean_threshold = float(mean_threshold)
        self.iterations = it
        self.padding = str(padding)
        self.clamp = bool(clamp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        y = x
        for _ in range(int(self.iterations)):
            # --- Rows
            row_mean = y.mean(dim=-1, keepdim=True)  # (B,C,H,1)
            row_var = y.var(dim=-1, keepdim=True, unbiased=False)  # (B,C,H,1)
            row_med = _median_filter_h(row_mean, k=int(self.window), padding="replicate")
            row_bad = (row_var < float(self.var_threshold)) & ((row_mean - row_med).abs() > float(self.mean_threshold))
            if bool(row_bad.any().item()):
                prev = torch.roll(y, shifts=1, dims=-2)
                nxt = torch.roll(y, shifts=-1, dims=-2)
                rep = 0.5 * (prev + nxt)
                y = torch.where(row_bad.expand_as(y), rep, y)

            # --- Cols
            col_mean = y.mean(dim=-2, keepdim=True)  # (B,C,1,W)
            col_var = y.var(dim=-2, keepdim=True, unbiased=False)  # (B,C,1,W)
            col_med = _median_filter_w(col_mean, k=int(self.window), padding="replicate")
            col_bad = (col_var < float(self.var_threshold)) & ((col_mean - col_med).abs() > float(self.mean_threshold))
            if bool(col_bad.any().item()):
                left = torch.roll(y, shifts=1, dims=-1)
                right = torch.roll(y, shifts=-1, dims=-1)
                rep = 0.5 * (left + right)
                y = torch.where(col_bad.expand_as(y), rep, y)

        return y.clamp(0.0, 1.0) if self.clamp else y


_VARIANTS: dict[str, dict] = {
    "line_defect_tiny": {"window": 7, "iters": 1},
    "line_defect_small": {"window": 9, "iters": 1},
    "line_defect_base": {"window": 11, "iters": 2},
}


def build_line_defect_corrector_denoiser(
    *,
    in_channels: int,  # unused
    sigma: float = 0.1,
    variant: str = "line_defect_small",
) -> nn.Module:
    _ = int(in_channels)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown LineDefectCorrector variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    mean_thr = max(0.15, min(0.7, 2.0 * float(sigma) + 0.1))
    return LineDefectCorrector(
        window=int(spec["window"]),
        var_threshold=1e-6,
        mean_threshold=float(mean_thr),
        iterations=int(spec["iters"]),
        padding="reflect",
        clamp=True,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    clean = torch.zeros(1, 1, 32, 32)
    clean[:, :, 10:22, 12:20] = 1.0
    noisy = clean.clone()
    # Inject a hot column and dead row.
    noisy[:, :, :, 7] = 1.0
    noisy[:, :, 15, :] = 0.0
    m = build_line_defect_corrector_denoiser(in_channels=1, sigma=0.1, variant="line_defect_tiny")
    out = m(noisy)
    print("line_defect_tiny", tuple(out.shape), float((out - clean).abs().mean().item()))

