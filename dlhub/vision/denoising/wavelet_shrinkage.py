
import math

import torch
from torch import nn
import torch.nn.functional as F

from ._utils import pad_to_multiple, unpad


def _haar_filters(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return 4 Haar analysis filters (LL, LH, HL, HH) as (4,1,2,2)."""

    s = 1.0 / math.sqrt(2.0)
    low = torch.tensor([s, s], device=device, dtype=dtype)
    high = torch.tensor([-s, s], device=device, dtype=dtype)

    ll = torch.outer(low, low)
    lh = torch.outer(low, high)
    hl = torch.outer(high, low)
    hh = torch.outer(high, high)
    return torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # (4,1,2,2)


def _soft_threshold(x: torch.Tensor, t: float) -> torch.Tensor:
    thr = float(t)
    if thr <= 0.0:
        return x
    return torch.sign(x) * F.relu(x.abs() - thr)


class WaveletShrinkage(nn.Module):
    """Haar wavelet shrinkage denoiser (torch-only, toy-first).

    Procedure:
    - Haar decomposition for `levels`
    - soft-threshold detail bands
    - inverse reconstruction

    Notes:
    - This is a classical baseline and uses no learnable parameters.
    - Implemented with grouped conv/conv_transpose so it's reasonably fast for small images.
    """

    def __init__(
        self,
        *,
        sigma: float = 0.1,
        levels: int = 2,
        thresh_mult: float = 2.8,
        padding: str = "reflect",
        clamp: bool = True,
    ) -> None:
        super().__init__()
        sig = float(sigma)
        if sig < 0.0:
            raise ValueError("sigma must be >= 0")
        lv = int(levels)
        if lv <= 0:
            raise ValueError("levels must be > 0")
        tm = float(thresh_mult)
        if tm <= 0.0:
            raise ValueError("thresh_mult must be > 0")
        self.sigma = sig
        self.levels = lv
        self.thresh_mult = tm
        self.padding = str(padding)
        self.clamp = bool(clamp)

        self.register_buffer("_filt", torch.empty(0), persistent=False)

    def _get_filters(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._filt.numel() == 0 or self._filt.device != device or self._filt.dtype != dtype:
            self._filt = _haar_filters(device=device, dtype=dtype)
        return self._filt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        # Pad to multiple of 2^levels.
        mult = 2 ** int(self.levels)
        x_pad, pad_hw = pad_to_multiple(x, mult, mode=self.padding)

        b, c, _, _ = x_pad.shape
        filt = self._get_filters(device=x_pad.device, dtype=x_pad.dtype)  # (4,1,2,2)

        # Expand to grouped conv weights: (4*C,1,2,2)
        w = filt.repeat(c, 1, 1, 1)  # (4C,1,2,2), ordered by channel groups

        ll = x_pad
        details: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for _ in range(int(self.levels)):
            # analysis
            y = F.conv2d(ll, w, bias=None, stride=2, padding=0, groups=c)  # (B,4C,H/2,W/2)
            y = y.view(b, c, 4, y.shape[-2], y.shape[-1])
            ll = y[:, :, 0]
            lh = _soft_threshold(y[:, :, 1], self.thresh_mult * self.sigma)
            hl = _soft_threshold(y[:, :, 2], self.thresh_mult * self.sigma)
            hh = _soft_threshold(y[:, :, 3], self.thresh_mult * self.sigma)
            details.append((lh, hl, hh))

        # reconstruction (use conv_transpose2d)
        for (lh, hl, hh) in reversed(details):
            y = torch.stack([ll, lh, hl, hh], dim=2)  # (B,C,4,H,W)
            y = y.view(b, 4 * c, y.shape[-2], y.shape[-1])  # (B,4C,H,W)
            ll = F.conv_transpose2d(y, w, bias=None, stride=2, padding=0, groups=c)  # (B,C,2H,2W)

        out = unpad(ll, pad_hw)
        return out.clamp(0.0, 1.0) if self.clamp else out


_VARIANTS: dict[str, dict] = {
    "wavelet_tiny": {"levels": 1, "thresh_mult": 2.2},
    "wavelet_small": {"levels": 2, "thresh_mult": 2.8},
    "wavelet_base": {"levels": 3, "thresh_mult": 3.0},
}


def build_wavelet_shrinkage_denoiser(
    *,
    in_channels: int,  # unused (kept for consistent signatures)
    sigma: float = 0.1,
    variant: str = "wavelet_small",
) -> nn.Module:
    _ = int(in_channels)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown WaveletShrinkage variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return WaveletShrinkage(
        sigma=float(sigma),
        levels=int(spec["levels"]),
        thresh_mult=float(spec["thresh_mult"]),
        padding="reflect",
        clamp=True,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.12).clamp(0.0, 1.0)
    m = build_wavelet_shrinkage_denoiser(in_channels=1, sigma=0.12, variant="wavelet_small")
    y = m(noisy)
    print("wavelet_small", tuple(y.shape), float((y - x).pow(2).mean().item()))

