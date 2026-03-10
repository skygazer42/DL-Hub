import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, scale_channels


class DynamicPointwiseConv(nn.Module):
    """Per-sample dynamic 1x1 conv generated from global context."""

    def __init__(self, in_ch: int, out_ch: int, *, hidden: int = 128) -> None:
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        h = int(hidden)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.in_ch, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, self.out_ch * self.in_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        b, c, h, w = x.shape
        if c != self.in_ch:
            raise ValueError(f"Expected in_ch={self.in_ch}, got {c}")
        w_dyn = self.mlp(self.pool(x)).view(b, self.out_ch, self.in_ch)  # (B,O,I)
        x_flat = x.view(b, c, h * w)
        y = torch.bmm(w_dyn, x_flat).view(b, self.out_ch, h, w)
        return y


class WeightNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1) -> None:
        super().__init__()
        self.pre = ConvBNAct(int(in_ch), int(out_ch), kernel_size=3, stride=int(stride), act="relu")
        self.dpw = DynamicPointwiseConv(int(out_ch), int(out_ch), hidden=max(64, int(out_ch) // 2))
        self.bn = nn.BatchNorm2d(int(out_ch))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        y = self.dpw(x)
        y = self.bn(y)
        return self.act(x + y)


class WeightNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        channels: tuple[int, int, int, int] = (48, 96, 192, 384),
        depths: tuple[int, int, int, int] = (1, 1, 2, 1),
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        chs = tuple(
            scale_channels(int(c), float(width_mult), min_ch=16, divisor=8) for c in channels
        )
        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), chs[0], kernel_size=3, stride=2, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        def stage(in_ch: int, out_ch: int, depth: int, *, stride: int) -> nn.Sequential:
            blocks: list[nn.Module] = [WeightNetBlock(in_ch, out_ch, stride=int(stride))]
            for _ in range(int(depth) - 1):
                blocks.append(WeightNetBlock(out_ch, out_ch, stride=1))
            return nn.Sequential(*blocks)

        self.stage1 = stage(chs[0], chs[0], depths[0], stride=1)
        self.stage2 = stage(chs[0], chs[1], depths[1], stride=2)
        self.stage3 = stage(chs[1], chs[2], depths[2], stride=2)
        self.stage4 = stage(chs[2], chs[3], depths[3], stride=2)
        self.head = GlobalAvgPoolHead(chs[3], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "weightnet_tiny": {"channels": (40, 80, 160, 320), "depths": (1, 1, 2, 1)},
    "weightnet_base": {"channels": (48, 96, 192, 384), "depths": (1, 1, 2, 1)},
    "weightnet_deep": {"channels": (48, 96, 192, 384), "depths": (2, 2, 3, 2)},
}


def build_weightnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "weightnet_base",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown WeightNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return WeightNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        channels=tuple(map(int, spec["channels"])),
        depths=tuple(map(int, spec["depths"])),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_weightnet_classifier(
        in_channels=3, num_classes=10, variant="weightnet_base", width_mult=0.5
    )
    y = m(x)
    print("weightnet_base", tuple(y.shape))
