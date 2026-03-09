
import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, SqueezeExcite, make_divisible


class GhostModuleV2(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, ratio: int = 2, dw_kernel: int = 3, act: str = "relu") -> None:
        super().__init__()
        out_ch = int(out_ch)
        init_ch = int((out_ch + int(ratio) - 1) // int(ratio))
        new_ch = out_ch - init_ch
        self.primary = ConvBNAct(int(in_ch), init_ch, kernel_size=1, stride=1, padding=0, act=act)
        self.cheap = ConvBNAct(init_ch, new_ch, kernel_size=int(dw_kernel), stride=1, groups=init_ch, act=act)
        self.out_ch = out_ch
        # light gate to modulate ghost features (v2-ish)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(init_ch, new_ch, kernel_size=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary(x)
        x2 = self.cheap(x1)
        if x2.numel() > 0:
            x2 = x2 * self.gate(x1)
        x = torch.cat([x1, x2], dim=1)
        return x[:, : self.out_ch]


class GhostBottleneckV2(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, *, stride: int, se: bool) -> None:
        super().__init__()
        self.stride = int(stride)
        self.ghost1 = GhostModuleV2(int(in_ch), int(mid_ch), act="relu")
        self.dw: nn.Module
        if self.stride > 1:
            self.dw = nn.Sequential(
                nn.Conv2d(int(mid_ch), int(mid_ch), kernel_size=3, stride=self.stride, padding=1, groups=int(mid_ch), bias=False),
                nn.BatchNorm2d(int(mid_ch)),
            )
        else:
            self.dw = nn.Identity()
        self.se = SqueezeExcite(int(mid_ch), se_ratio=0.25) if bool(se) else nn.Identity()
        self.ghost2 = GhostModuleV2(int(mid_ch), int(out_ch), act="relu")

        self.short: nn.Module
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


class GhostNetV2Classifier(nn.Module):
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
            # in, mid, out, stride, se
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
            blocks.append(GhostBottleneckV2(in_ch, mid, out, stride=int(s), se=bool(se)))
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
    "ghostnet_v2_0_5": {"w": 0.5},
    "ghostnet_v2_1_0": {"w": 1.0},
    "ghostnet_v2_1_3": {"w": 1.3},
}


def build_ghostnet_v2_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "ghostnet_v2_1_0",
    dropout: float = 0.2,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown GhostNetV2 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return GhostNetV2Classifier(in_channels=int(in_channels), num_classes=int(num_classes), width_mult=float(spec["w"]), dropout=float(dropout))


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_ghostnet_v2_classifier(in_channels=3, num_classes=10, variant="ghostnet_v2_0_5")
    y = m(x)
    print("ghostnet_v2_0_5", tuple(y.shape))

