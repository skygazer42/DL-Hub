
import torch
from torch import nn

from dlhub.vision.backbones._blocks import GlobalAvgPoolHead, LayerNorm2d


class FFTMix(nn.Module):
    """FFT-based token mixer (simplified, fixed H/W)."""

    def __init__(self, channels: int, *, h: int, w: int) -> None:
        super().__init__()
        c = int(channels)
        self.h = int(h)
        self.w = int(w)
        # Complex weights for rfft2 output: (H, W//2+1)
        self.weight = nn.Parameter(torch.randn(1, c, self.h, self.w // 2 + 1, 2) * 0.02)
        self.proj = nn.Conv2d(c, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h != self.h or w != self.w:
            raise ValueError("FFTMix expects fixed H/W")
        xf = torch.fft.rfft2(x, norm="ortho")
        w = torch.view_as_complex(self.weight)
        yf = xf * w
        y = torch.fft.irfft2(yf, s=(h, w), norm="ortho")
        return self.proj(y)


class FFTMLPBlock(nn.Module):
    def __init__(self, dim: int, *, h: int, w: int) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = LayerNorm2d(d)
        self.mix = FFTMix(d, h=int(h), w=int(w))
        self.norm2 = LayerNorm2d(d)
        self.mlp = nn.Sequential(nn.Conv2d(d, 4 * d, kernel_size=1), nn.GELU(), nn.Conv2d(4 * d, d, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mix(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FFTMLPClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int = 64,
        patch_size: int = 4,
        dim: int = 192,
        depth: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d = int(dim)
        p = int(patch_size)
        h = int(image_size) // p
        w = int(image_size) // p
        self.patch = nn.Sequential(nn.Conv2d(int(in_channels), d, kernel_size=p, stride=p), LayerNorm2d(d))
        self.blocks = nn.Sequential(*[FFTMLPBlock(d, h=h, w=w) for _ in range(int(depth))])
        self.head = GlobalAvgPoolHead(d, int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch(x)
        x = self.blocks(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "fft_mlp_tiny": {"dim": 192, "depth": 10, "patch": 4},
    "fft_mlp_small": {"dim": 256, "depth": 12, "patch": 4},
}


def build_fft_mlp_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "fft_mlp_tiny",
    image_size: int = 64,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown FFT-MLP variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return FFTMLPClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        patch_size=int(spec["patch"]),
        dim=int(spec["dim"]),
        depth=int(spec["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 3, 64, 64)
    m = build_fft_mlp_classifier(in_channels=3, num_classes=10, variant="fft_mlp_tiny", image_size=64)
    y = m(x)
    print("fft_mlp_tiny", tuple(y.shape))

