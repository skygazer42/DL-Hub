import torch
import torch.nn.functional as F
from torch import nn


def _box_filter(x: torch.Tensor, *, k: int, padding: str) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW, got {tuple(x.shape)}")
    k = int(k)
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be odd and >= 1")
    if k == 1:
        return x

    p = k // 2
    _, c, _, _ = x.shape
    weight = torch.ones((c, 1, k, k), device=x.device, dtype=x.dtype) / float(k * k)
    x_pad = F.pad(x, (p, p, p, p), mode=str(padding))
    return F.conv2d(x_pad, weight, bias=None, stride=1, padding=0, groups=c)


class LeeFilter(nn.Module):
    """Lee filter baseline for multiplicative (speckle-like) noise (torch-only, toy-first).

    For a speckle model:
        I = S * (1 + n),  with Var(n) = sigma^2
    A common local-linear estimator is:
        y = mean + k * (I - mean)
    with a weight k that decreases when observed variance can be explained by speckle noise.
    """

    def __init__(
        self,
        *,
        sigma: float = 0.15,
        window: int = 7,
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

        if float(self.sigma) == 0.0 or int(self.window) == 1:
            return x.clamp(0.0, 1.0) if self.clamp else x

        mean = _box_filter(x, k=int(self.window), padding=self.padding)
        mean2 = _box_filter(x * x, k=int(self.window), padding=self.padding)
        var = (mean2 - mean * mean).clamp_min(0.0)

        # Speckle noise variance scales with mean^2 (since I = S*(1+n)).
        cn2 = float(self.sigma) ** 2
        noise_var = cn2 * mean * mean

        k = (var - noise_var).clamp_min(0.0) / (var + float(self.eps))
        y = mean + k * (x - mean)
        return y.clamp(0.0, 1.0) if self.clamp else y


_VARIANTS: dict[str, dict] = {
    "lee_tiny": {"window": 3},
    "lee_small": {"window": 5},
    "lee_base": {"window": 7},
}


def build_lee_filter_denoiser(
    *,
    in_channels: int,  # unused (kept for consistent signatures)
    sigma: float = 0.15,
    variant: str = "lee_small",
) -> nn.Module:
    _ = int(in_channels)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown LeeFilter variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return LeeFilter(sigma=float(sigma), window=int(spec["window"]), padding="reflect", clamp=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    clean = torch.rand(2, 1, 64, 64)
    speckle_std = 0.15
    noisy = (clean + clean * torch.randn_like(clean) * speckle_std).clamp(0.0, 1.0)
    m = build_lee_filter_denoiser(in_channels=1, sigma=speckle_std, variant="lee_tiny")
    out = m(noisy)
    print("lee_tiny", tuple(out.shape), float((out - clean).pow(2).mean().item()))
