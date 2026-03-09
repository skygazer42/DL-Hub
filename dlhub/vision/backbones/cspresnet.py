
import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, DropPath, GlobalAvgPoolHead, scale_channels


class CSPBottleneck(nn.Module):
    def __init__(self, channels: int, *, hidden_ratio: float = 0.5, drop_path: float = 0.0) -> None:
        super().__init__()
        c = int(channels)
        hidden = max(8, int(round(c * float(hidden_ratio))))
        self.conv1 = ConvBNAct(c, hidden, kernel_size=1, stride=1, padding=0, act="silu")
        self.conv2 = ConvBNAct(hidden, c, kernel_size=3, stride=1, act="silu")
        self.dp = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dp(self.conv2(self.conv1(x)))


class CSPStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, depth: int, *, drop_path: float = 0.0) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        d = int(depth)
        self.down = ConvBNAct(c_in, c_out, kernel_size=3, stride=2, act="silu")
        self.pre = ConvBNAct(c_out, c_out, kernel_size=1, stride=1, padding=0, act="silu")

        c1 = c_out // 2
        c2 = c_out - c1
        self.c1 = c1
        self.c2 = c2

        dp_rates = torch.linspace(0.0, float(drop_path), steps=max(1, d)).tolist()
        blocks = [CSPBottleneck(c2, hidden_ratio=0.5, drop_path=float(dp_rates[i])) for i in range(d)]
        self.blocks = nn.Sequential(*blocks)
        self.fuse = ConvBNAct(c_out, c_out, kernel_size=1, stride=1, padding=0, act="silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(self.down(x))
        x1, x2 = x[:, : self.c1], x[:, self.c1 :]
        x2 = self.blocks(x2)
        x = torch.cat([x1, x2], dim=1)
        return self.fuse(x)


class CSPResNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int, int] = (64, 128, 256, 512),
        depths: tuple[int, int, int, int] = (1, 2, 6, 2),
        width_mult: float = 1.0,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = tuple(scale_channels(int(d), float(width_mult), min_ch=16, divisor=8) for d in dims)
        depths = tuple(int(d) for d in depths)

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), dims[0], kernel_size=3, stride=2, act="silu"),
            ConvBNAct(dims[0], dims[0], kernel_size=3, stride=1, act="silu"),
        )

        total = sum(depths[1:])
        dp_rates = torch.linspace(0.0, float(drop_path), steps=max(1, total)).tolist()
        dp_iter = iter(dp_rates)

        self.stage1 = nn.Identity()
        self.stage2 = CSPStage(dims[0], dims[1], depths[1], drop_path=float(next(dp_iter, 0.0)))
        self.stage3 = CSPStage(dims[1], dims[2], depths[2], drop_path=float(next(dp_iter, 0.0)))
        self.stage4 = CSPStage(dims[2], dims[3], depths[3], drop_path=float(next(dp_iter, 0.0)))

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
    "cspresnet_tiny": {"dims": (48, 96, 192, 384), "depths": (1, 1, 3, 1)},
    "cspresnet_small": {"dims": (64, 128, 256, 512), "depths": (1, 2, 6, 2)},
    "cspresnet_base": {"dims": (80, 160, 320, 640), "depths": (1, 3, 9, 3)},
}


def build_cspresnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "cspresnet_small",
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CSPResNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return CSPResNetClassifier(
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
    m = build_cspresnet_classifier(in_channels=3, num_classes=10, variant="cspresnet_tiny", width_mult=0.5)
    y = m(x)
    print("cspresnet_tiny", tuple(y.shape))
