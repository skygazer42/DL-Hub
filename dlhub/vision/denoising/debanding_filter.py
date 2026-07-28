import torch
import torch.nn.functional as F
from torch import nn


def _gaussian_kernel1d(
    k: int, sigma: float, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    k = int(k)
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be odd and >= 1")
    s = float(sigma)
    if s <= 0.0:
        raise ValueError("sigma must be > 0")

    if k == 1:
        return torch.ones((1,), device=device, dtype=dtype)

    r = k // 2
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    w = torch.exp(-0.5 * (x * x) / (s * s))
    return w / w.sum().clamp_min(1e-12)


def _gaussian_blur(x: torch.Tensor, *, k: int, sigma: float, padding: str) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW, got {tuple(x.shape)}")
    if k == 1:
        return x

    _, c, _, _ = x.shape
    kern = _gaussian_kernel1d(k, sigma, device=x.device, dtype=x.dtype)

    # depthwise separable conv: horizontal then vertical
    w_h = kern.view(1, 1, 1, k).repeat(c, 1, 1, 1)
    w_v = kern.view(1, 1, k, 1).repeat(c, 1, 1, 1)
    p = k // 2

    x_pad = F.pad(x, (p, p, 0, 0), mode=str(padding))
    y = F.conv2d(x_pad, w_h, bias=None, stride=1, padding=0, groups=c)
    y_pad = F.pad(y, (0, 0, p, p), mode=str(padding))
    y = F.conv2d(y_pad, w_v, bias=None, stride=1, padding=0, groups=c)
    return y


class DebandingFilter(nn.Module):
    """Debanding / dequantization baseline (torch-only, compact-first).

    Strategy:
    - Estimate edges via simple gradients.
    - Apply light Gaussian smoothing only in flat regions (low gradient magnitude).

    Intended for `noise_type=quantization` and similar banding artifacts.
    """

    def __init__(
        self,
        *,
        kernel_size: int = 5,
        sigma: float = 1.0,
        grad_threshold: float = 0.02,
        iterations: int = 1,
        padding: str = "reflect",
        clamp: bool = True,
    ) -> None:
        super().__init__()
        k = int(kernel_size)
        if k < 1 or k % 2 == 0:
            raise ValueError("kernel_size must be odd and >= 1")
        if float(sigma) <= 0.0:
            raise ValueError("sigma must be > 0")
        t = float(grad_threshold)
        if t <= 0.0:
            raise ValueError("grad_threshold must be > 0")
        it = int(iterations)
        if it <= 0:
            raise ValueError("iterations must be > 0")
        self.kernel_size = k
        self.sigma = float(sigma)
        self.grad_threshold = t
        self.iterations = it
        self.padding = str(padding)
        self.clamp = bool(clamp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        y = x
        for _ in range(int(self.iterations)):
            # Gradient magnitude (simple forward diffs).
            dx = y[..., :, 1:] - y[..., :, :-1]
            dy = y[..., 1:, :] - y[..., :-1, :]
            # Pad back to HxW
            dx = F.pad(dx, (0, 1, 0, 0))
            dy = F.pad(dy, (0, 0, 0, 1))
            grad = (dx.abs() + dy.abs()).mean(dim=1, keepdim=True)  # (B,1,H,W)

            smooth = _gaussian_blur(
                y, k=int(self.kernel_size), sigma=float(self.sigma), padding=self.padding
            )
            mask = grad < float(self.grad_threshold)
            y = torch.where(mask, smooth, y)

        return y.clamp(0.0, 1.0) if self.clamp else y


_VARIANTS: dict[str, dict] = {
    "deband_tiny": {"k": 3, "sigma": 0.9, "thr": 0.02, "iters": 1},
    "deband_small": {"k": 5, "sigma": 1.1, "thr": 0.02, "iters": 1},
    "deband_strong": {"k": 5, "sigma": 1.2, "thr": 0.04, "iters": 2},
}


def build_debanding_filter_denoiser(
    *,
    in_channels: int,  # unused
    sigma: float = 0.1,
    variant: str = "deband_small",
) -> nn.Module:
    _ = int(in_channels)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown DebandingFilter variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]

    # Heuristic: with higher noise sigma, allow smoothing in slightly higher-gradient regions.
    thr = float(spec["thr"]) * (1.0 + 0.5 * float(sigma))
    return DebandingFilter(
        kernel_size=int(spec["k"]),
        sigma=float(spec["sigma"]),
        grad_threshold=float(thr),
        iterations=int(spec["iters"]),
        padding="reflect",
        clamp=True,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    # Make a synthetic smooth ramp (shows banding after quantization).
    h, w = 64, 64
    ramp = torch.linspace(0.0, 1.0, w).view(1, 1, 1, w).expand(1, 1, h, w)
    bits = 5
    levels = float((1 << bits) - 1)
    noisy = torch.round(ramp * levels) / levels
    m = build_debanding_filter_denoiser(in_channels=1, sigma=0.1, variant="deband_tiny")
    out = m(noisy)
    print("deband_tiny", tuple(out.shape), float((out - ramp).abs().mean().item()))
