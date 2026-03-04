from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, DropPath, LayerNorm2d, scale_channels
from dlhub.vision.backbones._transformer import TransformerEncoderBlock


class EfficientViTV2ConvBlock(nn.Module):
    def __init__(self, dim: int, *, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        self.norm = LayerNorm2d(d)
        self.dw = nn.Conv2d(d, d, kernel_size=3, padding=1, groups=d, bias=False)
        self.bn = nn.BatchNorm2d(d)
        self.pw = nn.Conv2d(d, d, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.dp = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw(self.act(self.bn(self.dw(self.norm(x)))))
        y = self.dp(y)
        return x + y


class EfficientViTV2Classifier(nn.Module):
    """EfficientViT-V2-ish hybrid: conv stages + final transformer stage (simplified)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int] = (64, 128, 256),
        depths: tuple[int, int, int] = (2, 2, 4),
        heads: int = 8,
        width_mult: float = 1.0,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = tuple(scale_channels(int(d), float(width_mult), min_ch=16, divisor=8) for d in dims)
        depths = tuple(int(d) for d in depths)
        total = sum(depths)
        dp_rates = torch.linspace(0.0, float(drop_path), steps=max(1, total)).tolist()
        dp_iter = iter(dp_rates)

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), dims[0], kernel_size=3, stride=2, act="silu"),
            ConvBNAct(dims[0], dims[0], kernel_size=3, stride=1, act="silu"),
        )

        self.stage1 = nn.Sequential(*[EfficientViTV2ConvBlock(dims[0], drop_path=float(next(dp_iter, 0.0))) for _ in range(depths[0])])
        self.down2 = ConvBNAct(dims[0], dims[1], kernel_size=3, stride=2, act="silu")
        self.stage2 = nn.Sequential(*[EfficientViTV2ConvBlock(dims[1], drop_path=float(next(dp_iter, 0.0))) for _ in range(depths[1])])

        self.down3 = ConvBNAct(dims[1], dims[2], kernel_size=3, stride=2, act="silu")
        self.blocks = nn.Sequential(
            *[
                TransformerEncoderBlock(dims[2], int(heads), mlp_ratio=4.0, dropout=float(dropout), drop_path=float(next(dp_iter, 0.0)))
                for _ in range(depths[2])
            ]
        )
        self.norm = nn.LayerNorm(dims[2])
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(dims[2], int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down2(x)
        x = self.stage2(x)
        x = self.down3(x)
        # flatten to tokens for the transformer tail
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2).contiguous()
        t = self.blocks(t)
        t = self.norm(t)
        t = self.drop(t.mean(dim=1))
        return self.head(t)


_VARIANTS: dict[str, dict] = {
    "efficientvit_v2_tiny": {"dims": (64, 128, 256), "depths": (2, 2, 2), "heads": 8},
    "efficientvit_v2_small": {"dims": (64, 160, 320), "depths": (2, 2, 4), "heads": 8},
    "efficientvit_v2_base": {"dims": (80, 192, 384), "depths": (3, 3, 6), "heads": 12},
}


def build_efficientvit_v2_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "efficientvit_v2_tiny",
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown EfficientViT-V2 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return EfficientViTV2Classifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dims=tuple(map(int, spec["dims"])),
        depths=tuple(map(int, spec["depths"])),
        heads=int(spec["heads"]),
        width_mult=float(width_mult),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_efficientvit_v2_classifier(in_channels=3, num_classes=10, variant="efficientvit_v2_tiny", width_mult=0.5)
    y = m(x)
    print("efficientvit_v2_tiny", tuple(y.shape))
