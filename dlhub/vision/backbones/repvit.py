from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath, GlobalAvgPoolHead, LayerNorm2d, scale_channels


class RepDWConv(nn.Module):
    """Rep-style depthwise conv: 3x3 + 1x1 (+ identity) branches summed."""

    def __init__(self, channels: int, *, stride: int = 1) -> None:
        super().__init__()
        c = int(channels)
        s = int(stride)
        self.dw3 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=s, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
        )
        self.dw1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, stride=s, padding=0, groups=c, bias=False),
            nn.BatchNorm2d(c),
        )
        self.use_id = s == 1
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw3(x) + self.dw1(x)
        if self.use_id and y.shape == x.shape:
            y = y + x
        return self.act(y)


class RepViTBlock(nn.Module):
    def __init__(self, dim: int, *, mlp_ratio: float = 4.0, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        hidden = int(round(d * float(mlp_ratio)))
        self.norm1 = LayerNorm2d(d)
        self.mixer = RepDWConv(d, stride=1)
        self.dp1 = DropPath(float(drop_path))
        self.norm2 = LayerNorm2d(d)
        self.mlp = nn.Sequential(
            nn.Conv2d(d, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, d, kernel_size=1),
        )
        self.dp2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.mixer(self.norm1(x)))
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


class RepViTClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int, int],
        depths: tuple[int, int, int, int],
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
            nn.Conv2d(int(in_channels), dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.SiLU(inplace=True),
        )

        def make_stage(in_ch: int, out_ch: int, depth: int, *, stride: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            if int(stride) == 2:
                layers.append(RepDWConv(in_ch, stride=2))
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.SiLU(inplace=True))
            for _ in range(int(depth)):
                layers.append(RepViTBlock(out_ch, mlp_ratio=4.0, drop_path=float(next(dp_iter, 0.0))))
            return nn.Sequential(*layers)

        self.stage1 = make_stage(dims[0], dims[0], depths[0], stride=1)
        self.stage2 = make_stage(dims[0], dims[1], depths[1], stride=2)
        self.stage3 = make_stage(dims[1], dims[2], depths[2], stride=2)
        self.stage4 = make_stage(dims[2], dims[3], depths[3], stride=2)

        self.head = GlobalAvgPoolHead(dims[3], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "repvit_tiny": {"dims": (64, 128, 256, 512), "depths": (1, 2, 4, 2)},
    "repvit_small": {"dims": (64, 160, 320, 640), "depths": (2, 2, 6, 2)},
    "repvit_base": {"dims": (80, 192, 384, 768), "depths": (2, 3, 8, 3)},
}


def build_repvit_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "repvit_tiny",
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown RepViT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return RepViTClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dims=tuple(map(int, spec["dims"])),
        depths=tuple(map(int, spec["depths"])),
        width_mult=float(width_mult),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_repvit_classifier(in_channels=3, num_classes=10, variant="repvit_tiny", width_mult=0.5)
    y = m(x)
    print("repvit_tiny", tuple(y.shape))
