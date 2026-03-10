import torch
from torch import nn

from dlhub.vision.backbones._blocks import GlobalAvgPoolHead, LayerNorm2d, scale_channels


class VisionPermutator(nn.Module):
    """Vision Permutator (ViP) token mixer (simplified).

    Applies linear mixing along height, width, and channels and sums them.
    This implementation assumes a fixed spatial size after patch embedding.
    """

    def __init__(self, dim: int, *, h: int, w: int) -> None:
        super().__init__()
        d = int(dim)
        self.h = int(h)
        self.w = int(w)
        self.mlp_h = nn.Linear(self.h, self.h, bias=True)
        self.mlp_w = nn.Linear(self.w, self.w, bias=True)
        self.mlp_c = nn.Linear(d, d, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        if h != self.h or w != self.w:
            raise ValueError(f"Expected H,W={self.h},{self.w}, got {h},{w}")
        x_hw = x.permute(0, 2, 3, 1)  # (B,H,W,C)
        # mix along H
        y_h = self.mlp_h(x_hw.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)  # (B,H,W,C)
        # mix along W
        y_w = self.mlp_w(x_hw.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        # mix along C
        y_c = self.mlp_c(x_hw)
        y = (y_h + y_w + y_c) / 3.0
        return y.permute(0, 3, 1, 2).contiguous()


class ViPBlock(nn.Module):
    def __init__(self, dim: int, *, h: int, w: int) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = LayerNorm2d(d)
        self.mix = VisionPermutator(d, h=int(h), w=int(w))
        self.norm2 = LayerNorm2d(d)
        self.mlp = nn.Sequential(
            nn.Conv2d(d, 4 * d, kernel_size=1), nn.GELU(), nn.Conv2d(4 * d, d, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mix(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViPClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int = 64,
        patch_size: int = 4,
        dim: int = 192,
        depth: int = 10,
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d = scale_channels(int(dim), float(width_mult), min_ch=16, divisor=8)
        p = int(patch_size)
        h = int(image_size) // p
        w = int(image_size) // p
        self.patch = nn.Sequential(
            nn.Conv2d(int(in_channels), d, kernel_size=p, stride=p), LayerNorm2d(d)
        )
        self.blocks = nn.Sequential(*[ViPBlock(d, h=h, w=w) for _ in range(int(depth))])
        self.head = GlobalAvgPoolHead(d, int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch(x)
        x = self.blocks(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "vip_tiny": {"dim": 192, "depth": 10, "patch": 4},
    "vip_small": {"dim": 256, "depth": 12, "patch": 4},
}


def build_vip_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "vip_tiny",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ViP variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return ViPClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        patch_size=int(spec["patch"]),
        dim=int(spec["dim"]),
        depth=int(spec["depth"]),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_vip_classifier(
        in_channels=3, num_classes=10, variant="vip_tiny", image_size=64, width_mult=0.5
    )
    y = m(x)
    print("vip_tiny", tuple(y.shape))
