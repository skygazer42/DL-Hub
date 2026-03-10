import torch
import torch.nn.functional as F
from torch import nn


def _box_filter(x: torch.Tensor, *, k: int, padding: str) -> torch.Tensor:
    """Depthwise box filter over NCHW tensor."""

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


class AnscombeWiener(nn.Module):
    """Poisson-ish denoiser: Anscombe VST + local Wiener in VST domain (toy-first).

    This is a simple variance-stabilizing approach for Poisson-like noise:
    - Assume normalized image x in [0,1] represents counts / peak.
    - Use `sigma` to infer peak approximately: peak ~= 1 / sigma^2.
    - Apply Anscombe transform: z = 2 * sqrt(counts + 3/8).
    - Denoise z with local Wiener assuming unit noise variance.
    - Invert transform approximately and rescale back to [0,1].

    Notes:
    - This is a baseline meant for toy datasets; the inverse is a common approximation.
    - When sigma is small, peak becomes large and this approaches an identity transform.
    """

    def __init__(
        self,
        *,
        sigma: float = 0.18,
        window: int = 5,
        eps: float = 1e-6,
        padding: str = "reflect",
        clamp: bool = True,
    ) -> None:
        super().__init__()
        sig = float(sigma)
        if sig < 0.0:
            raise ValueError("sigma must be >= 0")
        w = int(window)
        if w < 1 or w % 2 == 0:
            raise ValueError("window must be odd and >= 1")
        self.sigma = sig
        self.window = w
        self.eps = float(eps)
        self.padding = str(padding)
        self.clamp = bool(clamp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        if float(self.sigma) == 0.0:
            return x.clamp(0.0, 1.0) if self.clamp else x

        peak = 1.0 / (float(self.sigma) * float(self.sigma))
        counts = (x.clamp_min(0.0) * float(peak)).to(torch.float32)
        z = 2.0 * torch.sqrt(counts + 0.375)  # Anscombe

        mean = _box_filter(z, k=int(self.window), padding=self.padding)
        mean2 = _box_filter(z * z, k=int(self.window), padding=self.padding)
        var = (mean2 - mean * mean).clamp_min(0.0)

        # In Anscombe domain, variance is approximately 1.
        noise_var = 1.0
        k = (var - noise_var).clamp_min(0.0) / (var + float(self.eps))
        z_hat = mean + k * (z - mean)

        # Approx inverse Anscombe
        counts_hat = (z_hat * 0.5).pow(2) - 0.375
        y = counts_hat / float(peak)
        return y.clamp(0.0, 1.0) if self.clamp else y


_VARIANTS: dict[str, dict] = {
    "anscombe_wiener_tiny": {"window": 3},
    "anscombe_wiener_small": {"window": 5},
    "anscombe_wiener_base": {"window": 7},
}


def build_anscombe_wiener_denoiser(
    *,
    in_channels: int,  # unused (kept for consistent signatures)
    sigma: float = 0.18,
    variant: str = "anscombe_wiener_small",
) -> nn.Module:
    _ = int(in_channels)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown AnscombeWiener variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    return AnscombeWiener(
        sigma=float(sigma), window=int(spec["window"]), padding="reflect", clamp=True
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    # Simulate Poisson-ish noise by quantizing counts.
    peak = 30.0
    clean = torch.rand(2, 1, 64, 64)
    counts = torch.poisson(clean * peak) / peak
    noisy = counts.clamp(0.0, 1.0)

    m = build_anscombe_wiener_denoiser(
        in_channels=1, sigma=1.0 / (peak**0.5), variant="anscombe_wiener_tiny"
    )
    out = m(noisy)
    print("anscombe_wiener_tiny", tuple(out.shape), float((out - clean).pow(2).mean().item()))
