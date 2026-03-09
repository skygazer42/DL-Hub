
import torch
from torch import nn
import torch.nn.functional as F


class _RecursiveResBlock(nn.Module):
    """Recursive residual block (weights shared across recursions)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        if c <= 0:
            raise ValueError("channels must be > 0")
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x), inplace=True)
        y = self.conv2(y)
        return F.relu(x + y, inplace=True)


class DRRN(nn.Module):
    """DRRN (Deep Recursive Residual Network), adapted for denoising (toy-first, pure torch).

    DRRN is originally used for SR/restoration. We keep resolution and use a recursive residual unit.
    Output is `x - residual`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        features: int = 64,
        recursions: int = 8,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        f = int(features)
        r = int(recursions)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if f < 8:
            raise ValueError("features must be >= 8")
        if r <= 0:
            raise ValueError("recursions must be > 0")

        self.in_conv = nn.Conv2d(c_in, f, kernel_size=3, padding=1, bias=True)
        self.ru = _RecursiveResBlock(f)
        self.out_conv = nn.Conv2d(f, c_in, kernel_size=3, padding=1, bias=True)
        self.recursions = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        y = F.relu(self.in_conv(x), inplace=True)
        for _ in range(self.recursions):
            y = self.ru(y)
        residual = self.out_conv(y)
        return x - residual


_VARIANTS: dict[str, dict] = {
    "drrn_tiny": {"features": 32, "recursions": 4},
    "drrn_small": {"features": 48, "recursions": 8},
    "drrn_base": {"features": 64, "recursions": 16},
}


def build_drrn_denoiser(*, in_channels: int, variant: str = "drrn_small") -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DRRN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return DRRN(in_channels=int(in_channels), features=int(spec["features"]), recursions=int(spec["recursions"]))


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_drrn_denoiser(in_channels=1, variant="drrn_tiny")
    y = m(noisy)
    print("drrn_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

