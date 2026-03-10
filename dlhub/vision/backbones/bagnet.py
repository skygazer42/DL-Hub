import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, scale_channels


class BagNetBlock(nn.Module):
    """BagNet-style block emphasizing small receptive fields (mostly 1x1 convs)."""

    def __init__(self, channels: int, *, use_3x3: bool = False, drop: float = 0.0) -> None:
        super().__init__()
        c = int(channels)
        if bool(use_3x3):
            self.conv = nn.Sequential(
                ConvBNAct(c, c, kernel_size=1, stride=1, act="relu"),
                ConvBNAct(c, c, kernel_size=3, stride=1, act="relu"),
                nn.Conv2d(c, c, kernel_size=1, bias=False),
                nn.BatchNorm2d(c),
            )
        else:
            self.conv = nn.Sequential(
                ConvBNAct(c, c, kernel_size=1, stride=1, act="relu"),
                ConvBNAct(c, c, kernel_size=1, stride=1, act="relu"),
                nn.Conv2d(c, c, kernel_size=1, bias=False),
                nn.BatchNorm2d(c),
            )
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=float(drop)) if float(drop) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.drop(y)
        return self.act(x + y)


class BagNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        depths: tuple[int, int, int, int] = (2, 2, 2, 2),
        use_3x3_every: int = 2,
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        chs = tuple(scale_channels(int(c), float(width_mult)) for c in channels)
        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), chs[0], kernel_size=3, stride=2, act="relu"),
            ConvBNAct(chs[0], chs[0], kernel_size=1, stride=1, padding=0, act="relu"),
        )
        self.stages = nn.ModuleList()
        in_ch = chs[0]
        for i, (out_ch, d) in enumerate(zip(chs, depths)):
            blocks: list[nn.Module] = []
            if i > 0:
                blocks.append(
                    ConvBNAct(in_ch, out_ch, kernel_size=1, stride=2, padding=0, act="relu")
                )
                in_ch = out_ch
            for j in range(int(d)):
                blocks.append(BagNetBlock(in_ch, use_3x3=((j % int(use_3x3_every)) == 0), drop=0.0))
            self.stages.append(nn.Sequential(*blocks))
        self.head = GlobalAvgPoolHead(chs[-1], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        for s in self.stages:
            x = s(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "bagnet9": {"channels": (64, 128, 256, 512), "depths": (2, 2, 2, 2), "use_3x3_every": 4},
    "bagnet17": {"channels": (64, 128, 256, 512), "depths": (2, 2, 3, 2), "use_3x3_every": 2},
    "bagnet33": {"channels": (64, 128, 256, 512), "depths": (3, 3, 4, 3), "use_3x3_every": 1},
}


def build_bagnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "bagnet17",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown BagNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return BagNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        channels=tuple(map(int, spec["channels"])),
        depths=tuple(map(int, spec["depths"])),
        use_3x3_every=int(spec["use_3x3_every"]),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["bagnet9", "bagnet17"]:
        m = build_bagnet_classifier(in_channels=3, num_classes=10, variant=v, width_mult=0.5)
        y = m(x)
        print(v, tuple(y.shape))
