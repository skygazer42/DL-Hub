
import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, DropPath, GlobalAvgPoolHead, LayerNorm2d, scale_channels
from dlhub.vision.backbones._transformer import MultiheadSelfAttention, MLP


class CMTBlock(nn.Module):
    """CMT block (simplified): local perception + attention + MLP."""

    def __init__(self, dim: int, num_heads: int, *, mlp_ratio: float = 4.0, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        self.lpu = nn.Conv2d(d, d, kernel_size=3, padding=1, groups=d, bias=False)
        self.lpu_bn = nn.BatchNorm2d(d)
        self.drop_path = DropPath(float(drop_path))

        self.norm1 = nn.LayerNorm(d)
        self.attn = MultiheadSelfAttention(d, int(num_heads), dropout=0.0)
        self.dp1 = DropPath(float(drop_path))

        self.norm2 = nn.LayerNorm(d)
        self.mlp = MLP(d, int(round(d * float(mlp_ratio))), dropout=0.0, act="gelu")
        self.dp2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # local perception
        x = x + self.drop_path(self.lpu_bn(self.lpu(x)))
        # attention
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        t = t + self.dp1(self.attn(self.norm1(t)))
        t = t + self.dp2(self.mlp(self.norm2(t)))
        return t.transpose(1, 2).contiguous().view(b, c, h, w)


class CMTClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int, int] = (64, 128, 256, 512),
        depths: tuple[int, int, int, int] = (2, 2, 6, 2),
        heads: tuple[int, int, int, int] = (2, 4, 8, 16),
        width_mult: float = 1.0,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = tuple(scale_channels(int(d), float(width_mult), min_ch=16, divisor=8) for d in dims)
        depths = tuple(int(x) for x in depths)
        heads = tuple(int(h) for h in heads)
        total = sum(depths)
        dp_rates = torch.linspace(0.0, float(drop_path), steps=total).tolist()
        dp_iter = iter(dp_rates)

        self.stem = nn.Sequential(
            nn.Conv2d(int(in_channels), dims[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.down = nn.ModuleList()
        self.down.append(nn.Identity())
        for i in range(3):
            self.down.append(nn.Sequential(nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2, bias=False), nn.BatchNorm2d(dims[i + 1])))

        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = [CMTBlock(dims[i], heads[i], drop_path=float(next(dp_iter))) for _ in range(depths[i])]
            self.stages.append(nn.Sequential(*blocks))

        self.head = GlobalAvgPoolHead(dims[-1], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        for i in range(4):
            x = self.down[i](x)
            x = self.stages[i](x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "cmt_tiny": {"dims": (48, 96, 192, 384), "depths": (2, 2, 4, 2), "heads": (2, 3, 6, 12)},
    "cmt_base": {"dims": (64, 128, 256, 512), "depths": (2, 2, 6, 2), "heads": (2, 4, 8, 16)},
}


def build_cmt_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "cmt_base",
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CMT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return CMTClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dims=tuple(map(int, spec["dims"])),
        depths=tuple(map(int, spec["depths"])),
        heads=tuple(map(int, spec["heads"])),
        width_mult=float(width_mult),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_cmt_classifier(in_channels=3, num_classes=10, variant="cmt_tiny", width_mult=0.5)
    y = m(x)
    print("cmt_tiny", tuple(y.shape))

