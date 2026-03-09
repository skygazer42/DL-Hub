
import torch
from torch import nn
import torch.nn.functional as F


class _ResBlockNoBN(nn.Module):
    """EDSR-style residual block (no batch-norm)."""

    def __init__(self, channels: int, *, res_scale: float = 0.1) -> None:
        super().__init__()
        c = int(channels)
        if c <= 0:
            raise ValueError("channels must be > 0")
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.res_scale = float(res_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        return x + y * self.res_scale


class EDSR(nn.Module):
    """EDSR (Enhanced Deep Super-Resolution Network) adapted for denoising.

    EDSR is a strong residual backbone without BN. For denoising we keep resolution
    and predict a residual/noise map, returning `x - residual`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        features: int = 64,
        num_blocks: int = 16,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        b = int(num_blocks)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f < 8:
            raise ValueError("features must be >= 8")
        if b <= 0:
            raise ValueError("num_blocks must be > 0")

        self.head = nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True)
        self.body = nn.Sequential(*[_ResBlockNoBN(f, res_scale=float(res_scale)) for _ in range(b)])
        self.body_tail = nn.Conv2d(f, f, kernel_size=3, padding=1, bias=True)
        self.tail = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        f0 = self.head(x)
        y = self.body(f0)
        y = self.body_tail(y)
        y = y + f0
        residual = self.tail(F.relu(y, inplace=True))
        return x - residual


_VARIANTS: dict[str, dict] = {
    "edsr_tiny": {"features": 32, "num_blocks": 4, "res_scale": 0.1},
    "edsr_small": {"features": 48, "num_blocks": 8, "res_scale": 0.1},
    "edsr_base": {"features": 64, "num_blocks": 16, "res_scale": 0.1},
}


def build_edsr_denoiser(*, in_channels: int, variant: str = "edsr_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown EDSR variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return EDSR(
        in_channels=int(in_channels),
        features=int(spec["features"]),
        num_blocks=int(spec["num_blocks"]),
        res_scale=float(spec["res_scale"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 3, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_edsr_denoiser(in_channels=3, variant="edsr_tiny")
    y = m(noisy)
    print("edsr_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

