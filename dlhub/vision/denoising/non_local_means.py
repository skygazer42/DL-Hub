
import torch
from torch import nn
import torch.nn.functional as F


def _shift2d(x: torch.Tensor, *, dy: int, dx: int) -> torch.Tensor:
    """Shift NCHW tensor by (dy, dx) with replicate padding."""

    if x.ndim != 4:
        raise ValueError(f"Expected NCHW, got {tuple(x.shape)}")
    dy = int(dy)
    dx = int(dx)
    if dy == 0 and dx == 0:
        return x

    b, c, h, w = x.shape
    top = max(0, dy)
    bottom = max(0, -dy)
    left = max(0, dx)
    right = max(0, -dx)

    x_pad = F.pad(x, (left, right, top, bottom), mode="replicate")
    y0 = bottom
    x0 = right
    return x_pad[:, :, y0 : y0 + h, x0 : x0 + w]


def _patch_ssd(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    patch_size: int,
    padding: str,
) -> torch.Tensor:
    """Per-pixel patch SSD between x and y, summed over channels (NCHW -> N1HW)."""

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if x.ndim != 4:
        raise ValueError(f"Expected NCHW, got {tuple(x.shape)}")

    k = int(patch_size)
    if k < 1 or k % 2 == 0:
        raise ValueError("patch_size must be odd and >= 1")

    diff2 = (x - y).pow(2)  # (B,C,H,W)
    if k == 1:
        return diff2.sum(dim=1, keepdim=True)

    p = k // 2
    b, c, _, _ = x.shape
    weight = torch.ones((c, 1, k, k), device=x.device, dtype=x.dtype)
    diff2_pad = F.pad(diff2, (p, p, p, p), mode=str(padding))
    ssd = F.conv2d(diff2_pad, weight, bias=None, stride=1, padding=0, groups=c)  # (B,C,H,W)
    return ssd.sum(dim=1, keepdim=True)  # (B,1,H,W)


class NonLocalMeans(nn.Module):
    """Non-local means (NLM) baseline (torch-only, toy-first).

    This is a simplified NLM implementation that:
    - uses patch SSD computed via depthwise conv
    - scans a square search window of radius `search_radius`

    Complexity: O((2R+1)^2 * H * W) per iteration, intended for small images.
    """

    def __init__(
        self,
        *,
        sigma: float = 0.1,
        patch_size: int = 3,
        search_radius: int = 3,
        h: float | None = None,
        iterations: int = 1,
        padding: str = "reflect",
        clamp: bool = True,
    ) -> None:
        super().__init__()
        sig = float(sigma)
        if sig < 0.0:
            raise ValueError("sigma must be >= 0")
        ps = int(patch_size)
        if ps < 1 or ps % 2 == 0:
            raise ValueError("patch_size must be odd and >= 1")
        sr = int(search_radius)
        if sr < 0:
            raise ValueError("search_radius must be >= 0")
        it = int(iterations)
        if it <= 0:
            raise ValueError("iterations must be > 0")

        self.sigma = sig
        self.patch_size = ps
        self.search_radius = sr
        self.h = float(h) if h is not None else max(1e-6, 1.2 * sig)
        self.iterations = it
        self.padding = str(padding)
        self.clamp = bool(clamp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        if self.search_radius == 0:
            return x.clamp(0.0, 1.0) if self.clamp else x

        sr = int(self.search_radius)
        ps = int(self.patch_size)
        h2 = float(self.h) * float(self.h)

        y = x
        for _ in range(int(self.iterations)):
            num = torch.zeros_like(y)
            den = torch.zeros_like(y[:, :1])

            for dy in range(-sr, sr + 1):
                for dx in range(-sr, sr + 1):
                    shifted = _shift2d(y, dy=dy, dx=dx)
                    dist = _patch_ssd(y, shifted, patch_size=ps, padding=self.padding)  # (B,1,H,W)
                    w = torch.exp(-dist / max(1e-12, h2))
                    num = num + w * shifted
                    den = den + w

            y = num / den.clamp_min(1e-12)

        return y.clamp(0.0, 1.0) if self.clamp else y


_VARIANTS: dict[str, dict] = {
    "nlm_fast": {"patch": 3, "search": 2, "iters": 1},
    "nlm_quality": {"patch": 3, "search": 3, "iters": 1},
    "nlm_strong": {"patch": 5, "search": 3, "iters": 1},
}


def build_non_local_means_denoiser(
    *,
    in_channels: int,  # unused (kept for consistent signatures)
    sigma: float = 0.1,
    variant: str = "nlm_quality",
) -> nn.Module:
    _ = int(in_channels)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown NLM variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return NonLocalMeans(
        sigma=float(sigma),
        patch_size=int(spec["patch"]),
        search_radius=int(spec["search"]),
        h=max(1e-6, 1.2 * float(sigma)),
        iterations=int(spec["iters"]),
        padding="reflect",
        clamp=True,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(1, 1, 48, 48)
    noisy = (x + torch.randn_like(x) * 0.12).clamp(0.0, 1.0)
    m = build_non_local_means_denoiser(in_channels=1, sigma=0.12, variant="nlm_fast")
    y = m(noisy)
    print("nlm_fast", tuple(y.shape), float((y - x).pow(2).mean().item()))

