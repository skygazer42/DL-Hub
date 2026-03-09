
import torch
from torch import nn
import torch.nn.functional as F


def _box_filter(x: torch.Tensor, *, k: int, padding: str) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW, got {tuple(x.shape)}")
    k = int(k)
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be odd and >= 1")
    if k == 1:
        return x

    p = k // 2
    b, c, _, _ = x.shape
    weight = torch.ones((c, 1, k, k), device=x.device, dtype=x.dtype) / float(k * k)
    x_pad = F.pad(x, (p, p, p, p), mode=str(padding))
    return F.conv2d(x_pad, weight, bias=None, stride=1, padding=0, groups=c)


class GuidedFilter(nn.Module):
    """Guided filter baseline (torch-only, toy-first).

    This implements a simple per-channel guided filter using the input image as guidance:
      q = mean_a * I + mean_b
    where a and b are estimated from local linear regression.

    Notes:
    - This version treats each channel independently (no cross-channel covariance).
    - It can preserve edges better than a plain box/gaussian blur.
    """

    def __init__(
        self,
        *,
        radius: int = 4,
        eps: float = 1e-3,
        padding: str = "reflect",
        clamp: bool = True,
    ) -> None:
        super().__init__()
        r = int(radius)
        if r < 0:
            raise ValueError("radius must be >= 0")
        if float(eps) <= 0.0:
            raise ValueError("eps must be > 0")
        self.radius = r
        self.eps = float(eps)
        self.padding = str(padding)
        self.clamp = bool(clamp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        k = 2 * int(self.radius) + 1
        if k == 1:
            return x.clamp(0.0, 1.0) if self.clamp else x

        I = x  # guidance
        p = x  # filtering input

        mean_I = _box_filter(I, k=k, padding=self.padding)
        mean_p = _box_filter(p, k=k, padding=self.padding)
        mean_Ip = _box_filter(I * p, k=k, padding=self.padding)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = _box_filter(I * I, k=k, padding=self.padding)
        var_I = (mean_II - mean_I * mean_I).clamp_min(0.0)

        a = cov_Ip / (var_I + float(self.eps))
        b = mean_p - a * mean_I

        mean_a = _box_filter(a, k=k, padding=self.padding)
        mean_b = _box_filter(b, k=k, padding=self.padding)
        q = mean_a * I + mean_b
        return q.clamp(0.0, 1.0) if self.clamp else q


_VARIANTS: dict[str, dict] = {
    "guided_filter_tiny": {"radius": 2, "eps": 1e-3},
    "guided_filter_small": {"radius": 4, "eps": 1e-3},
    "guided_filter_base": {"radius": 6, "eps": 2e-3},
}


def build_guided_filter_denoiser(
    *,
    in_channels: int,  # unused (kept for consistent signatures)
    sigma: float = 0.1,
    variant: str = "guided_filter_small",
) -> nn.Module:
    _ = int(in_channels)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown GuidedFilter variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    # Heuristic: scale eps with sigma (bigger noise -> stronger regularization).
    eps = float(spec["eps"]) + 0.25 * float(sigma) * float(sigma)
    return GuidedFilter(radius=int(spec["radius"]), eps=eps, padding="reflect", clamp=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.12).clamp(0.0, 1.0)
    m = build_guided_filter_denoiser(in_channels=1, sigma=0.12, variant="guided_filter_tiny")
    y = m(noisy)
    print("guided_filter_tiny", tuple(y.shape), float((y - x).pow(2).mean().item()))

