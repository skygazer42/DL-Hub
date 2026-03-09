
import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, DropPath, GlobalAvgPoolHead, LayerNorm2d, scale_channels


class PartialConv2d(nn.Module):
    """FasterNet PConv (simplified).

    Applies a 3x3 conv to a subset of channels and concatenates with untouched channels.
    """

    def __init__(self, channels: int, *, ratio: float = 0.25) -> None:
        super().__init__()
        c = int(channels)
        r = float(ratio)
        c_p = max(1, int(round(c * r)))
        self.c_p = c_p
        self.conv = nn.Conv2d(c_p, c_p, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, : self.c_p]
        x2 = x[:, self.c_p :]
        x1 = self.conv(x1)
        return torch.cat([x1, x2], dim=1)


class FasterNetBlock(nn.Module):
    def __init__(self, dim: int, *, ratio: float = 0.25, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        hidden = 4 * d
        self.norm1 = LayerNorm2d(d)
        self.pconv = PartialConv2d(d, ratio=float(ratio))
        self.dp1 = DropPath(float(drop_path))
        self.norm2 = LayerNorm2d(d)
        self.mlp = nn.Sequential(
            nn.Conv2d(d, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, d, kernel_size=1),
        )
        self.dp2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.pconv(self.norm1(x)))
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


class FasterNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int, int],
        depths: tuple[int, int, int, int],
        pconv_ratio: float = 0.25,
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
            ConvBNAct(int(in_channels), dims[0], kernel_size=3, stride=2, act="gelu"),
            ConvBNAct(dims[0], dims[0], kernel_size=3, stride=1, act="gelu"),
        )

        def make_stage(in_ch: int, out_ch: int, depth: int, *, stride: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            if int(stride) == 2:
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2, bias=True))
            for _ in range(int(depth)):
                layers.append(
                    FasterNetBlock(
                        out_ch,
                        ratio=float(pconv_ratio),
                        drop_path=float(next(dp_iter, 0.0)),
                    )
                )
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
    "fasternet_t0": {"dims": (48, 96, 192, 384), "depths": (1, 1, 3, 1), "ratio": 0.25},
    "fasternet_t1": {"dims": (64, 128, 256, 512), "depths": (1, 2, 4, 2), "ratio": 0.25},
    "fasternet_s": {"dims": (80, 160, 320, 640), "depths": (2, 2, 6, 2), "ratio": 0.25},
}


def build_fasternet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "fasternet_t0",
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown FasterNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return FasterNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dims=tuple(map(int, spec["dims"])),
        depths=tuple(map(int, spec["depths"])),
        pconv_ratio=float(spec["ratio"]),
        width_mult=float(width_mult),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_fasternet_classifier(in_channels=3, num_classes=10, variant="fasternet_t0", width_mult=0.5)
    y = m(x)
    print("fasternet_t0", tuple(y.shape))
