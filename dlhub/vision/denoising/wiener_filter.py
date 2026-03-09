
import torch
from torch import nn
import torch.nn.functional as F


def _box_filter(x: torch.Tensor, *, k: int, padding: str) -> torch.Tensor:
    """Depthwise box filter over NCHW tensor."""

    if x.ndim != 4:
        raise ValueError(f"Expected NCHW, got {tuple(x.shape)}")
    k = int(k)
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be odd and >= 1")
    p = k // 2
    if k == 1:
        return x

    b, c, _, _ = x.shape
    weight = torch.ones((c, 1, k, k), device=x.device, dtype=x.dtype) / float(k * k)
    x_pad = F.pad(x, (p, p, p, p), mode=str(padding))
    return F.conv2d(x_pad, weight, bias=None, stride=1, padding=0, groups=c)


class WienerFilter(nn.Module):
    """Local Wiener filter baseline (torch-only, toy-first).

    A simple local-statistics denoiser:
      y = mean + (max(0, var - noise_var) / (var + eps)) * (x - mean)
    """

    def __init__(
        self,
        *,
        sigma: float = 0.1,
        window: int = 5,
        eps: float = 1e-6,
        padding: str = "reflect",
        clamp: bool = True,
    ) -> None:
        super().__init__()
        if float(sigma) < 0.0:
            raise ValueError("sigma must be >= 0")
        w = int(window)
        if w < 1 or w % 2 == 0:
            raise ValueError("window must be odd and >= 1")
        self.sigma = float(sigma)
        self.window = w
        self.eps = float(eps)
        self.padding = str(padding)
        self.clamp = bool(clamp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        mean = _box_filter(x, k=int(self.window), padding=self.padding)
        mean2 = _box_filter(x * x, k=int(self.window), padding=self.padding)
        var = (mean2 - mean * mean).clamp_min(0.0)

        noise_var = float(self.sigma) ** 2
        k = (var - noise_var).clamp_min(0.0) / (var + float(self.eps))
        y = mean + k * (x - mean)
        return y.clamp(0.0, 1.0) if self.clamp else y


_VARIANTS: dict[str, dict] = {
    "wiener_tiny": {"window": 3},
    "wiener_small": {"window": 5},
    "wiener_base": {"window": 7},
}


def build_wiener_filter_denoiser(
    *,
    in_channels: int,  # unused (kept for consistent signatures)
    sigma: float = 0.1,
    variant: str = "wiener_small",
) -> nn.Module:
    _ = int(in_channels)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown WienerFilter variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return WienerFilter(sigma=float(sigma), window=int(spec["window"]), padding="reflect", clamp=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.12).clamp(0.0, 1.0)
    m = build_wiener_filter_denoiser(in_channels=1, sigma=0.12, variant="wiener_tiny")
    y = m(noisy)
    print("wiener_tiny", tuple(y.shape), float((y - x).pow(2).mean().item()))

