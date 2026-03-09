
import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, SqueezeExcite, make_divisible


class RepGhostModule(nn.Module):
    """RepGhost-style module (multi-branch conv + cheap depthwise, simplified)."""

    def __init__(self, in_ch: int, out_ch: int, *, ratio: int = 2, act: str = "relu") -> None:
        super().__init__()
        out_ch = int(out_ch)
        init_ch = int((out_ch + int(ratio) - 1) // int(ratio))
        new_ch = out_ch - init_ch
        self.branch3 = ConvBNAct(int(in_ch), init_ch, kernel_size=3, stride=1, act=act)
        self.branch1 = ConvBNAct(int(in_ch), init_ch, kernel_size=1, stride=1, padding=0, act=act)
        self.cheap = ConvBNAct(init_ch, new_ch, kernel_size=3, stride=1, groups=init_ch, act=act)
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = 0.5 * (self.branch3(x) + self.branch1(x))
        x2 = self.cheap(x1)
        x = torch.cat([x1, x2], dim=1)
        return x[:, : self.out_ch]


class RepGhostBottleneck(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, *, stride: int, se: bool) -> None:
        super().__init__()
        self.stride = int(stride)
        self.ghost1 = RepGhostModule(int(in_ch), int(mid_ch), act="relu")
        if self.stride > 1:
            self.dw = nn.Sequential(
                nn.Conv2d(int(mid_ch), int(mid_ch), kernel_size=3, stride=self.stride, padding=1, groups=int(mid_ch), bias=False),
                nn.BatchNorm2d(int(mid_ch)),
            )
        else:
            self.dw = nn.Identity()
        self.se = SqueezeExcite(int(mid_ch), se_ratio=0.25) if bool(se) else nn.Identity()
        self.ghost2 = RepGhostModule(int(mid_ch), int(out_ch), act="relu")

        if int(in_ch) == int(out_ch) and self.stride == 1:
            self.short = nn.Identity()
        else:
            self.short = nn.Sequential(
                nn.Conv2d(int(in_ch), int(in_ch), kernel_size=3, stride=self.stride, padding=1, groups=int(in_ch), bias=False),
                nn.BatchNorm2d(int(in_ch)),
                nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, bias=False),
                nn.BatchNorm2d(int(out_ch)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ghost1(x)
        y = self.dw(y)
        y = self.se(y)
        y = self.ghost2(y)
        return y + self.short(x)


class RepGhostNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width_mult: float = 1.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        w = float(width_mult)

        def c(ch: int) -> int:
            return make_divisible(int(round(int(ch) * w)), 8)

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), c(16), kernel_size=3, stride=2, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        cfg = [
            (c(16), c(32), c(16), 1, False),
            (c(16), c(48), c(24), 2, False),
            (c(24), c(72), c(24), 1, False),
            (c(24), c(72), c(40), 2, True),
            (c(40), c(120), c(40), 1, True),
            (c(40), c(240), c(80), 2, False),
            (c(80), c(200), c(80), 1, False),
            (c(80), c(184), c(112), 1, True),
            (c(112), c(480), c(160), 2, True),
        ]
        blocks: list[nn.Module] = []
        in_ch = c(16)
        for _, mid, out, s, se in cfg:
            blocks.append(RepGhostBottleneck(in_ch, mid, out, stride=int(s), se=bool(se)))
            in_ch = int(out)
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            ConvBNAct(in_ch, c(960), kernel_size=1, stride=1, padding=0, act="relu"),
            GlobalAvgPoolHead(c(960), int(num_classes), dropout=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "repghostnet_0_5": {"w": 0.5},
    "repghostnet_1_0": {"w": 1.0},
    "repghostnet_1_3": {"w": 1.3},
}


def build_repghostnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "repghostnet_1_0",
    dropout: float = 0.2,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown RepGhostNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return RepGhostNetClassifier(in_channels=int(in_channels), num_classes=int(num_classes), width_mult=float(spec["w"]), dropout=float(dropout))


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_repghostnet_classifier(in_channels=3, num_classes=10, variant="repghostnet_0_5")
    y = m(x)
    print("repghostnet_0_5", tuple(y.shape))

