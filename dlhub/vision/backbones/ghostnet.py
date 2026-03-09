
import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, SqueezeExcite, scale_channels


def _c(ch: int, width_mult: float) -> int:
    return scale_channels(int(ch), float(width_mult), min_ch=8, divisor=8)


def _cheap_depthwise(in_ch: int, out_ch: int, *, kernel_size: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            int(in_ch),
            int(out_ch),
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=int(kernel_size) // 2,
            groups=int(in_ch),
            bias=False,
        ),
        nn.BatchNorm2d(int(out_ch)),
        nn.ReLU(inplace=True),
    )


class GhostModule(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, ratio: int = 2) -> None:
        super().__init__()
        out_ch = int(out_ch)
        primary = int((out_ch + int(ratio) - 1) // int(ratio))
        cheap = int(out_ch - primary)
        self.primary = nn.Sequential(
            nn.Conv2d(int(in_ch), primary, kernel_size=1, bias=False),
            nn.BatchNorm2d(primary),
            nn.ReLU(inplace=True),
        )
        self.cheap = _cheap_depthwise(primary, cheap, kernel_size=3, stride=1) if cheap > 0 else nn.Identity()
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary(x)
        x2 = self.cheap(x1)
        out = torch.cat([x1, x2], dim=1) if isinstance(x2, torch.Tensor) else x1
        return out[:, : self.out_ch, :, :]


class GhostBottleneck(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int, *, stride: int, se_ratio: float) -> None:
        super().__init__()
        self.ghost1 = GhostModule(in_ch, mid_ch)
        self.dw = (
            nn.Sequential(
                nn.Conv2d(int(mid_ch), int(mid_ch), kernel_size=3, stride=int(stride), padding=1, groups=int(mid_ch), bias=False),
                nn.BatchNorm2d(int(mid_ch)),
            )
            if int(stride) != 1
            else nn.Identity()
        )
        self.se = SqueezeExcite(int(mid_ch), se_ratio=float(se_ratio)) if float(se_ratio) > 0 else nn.Identity()
        self.ghost2 = GhostModule(mid_ch, out_ch)

        if int(stride) == 1 and int(in_ch) == int(out_ch):
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(int(in_ch), int(in_ch), kernel_size=3, stride=int(stride), padding=1, groups=int(in_ch), bias=False),
                nn.BatchNorm2d(int(in_ch)),
                nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, bias=False),
                nn.BatchNorm2d(int(out_ch)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ghost1(x)
        out = self.dw(out)
        out = self.se(out)
        out = self.ghost2(out)
        return out + self.shortcut(x)


class GhostNetClassifier(nn.Module):
    def __init__(self, *, in_channels: int, num_classes: int, width_mult: float, dropout: float, se_ratio: float) -> None:
        super().__init__()
        w = float(width_mult)

        self.stem = nn.Sequential(ConvBNAct(int(in_channels), _c(16, w), kernel_size=3, stride=2, act="relu"))

        cfg = [
            # (exp, out, se, stride)
            (16, 16, 0.0, 1),
            (48, 24, float(se_ratio), 2),
            (72, 24, 0.0, 1),
            (72, 40, float(se_ratio), 2),
            (120, 40, float(se_ratio), 1),
            (240, 80, 0.0, 2),
            (200, 80, 0.0, 1),
            (184, 80, 0.0, 1),
            (184, 80, 0.0, 1),
            (480, 112, float(se_ratio), 1),
            (672, 160, float(se_ratio), 2),
            (960, 160, float(se_ratio), 1),
        ]

        layers: list[nn.Module] = []
        in_ch = _c(16, w)
        for exp, out, se, s in cfg:
            layers.append(GhostBottleneck(in_ch, _c(exp, w), _c(out, w), stride=int(s), se_ratio=float(se)))
            in_ch = _c(out, w)
        self.features = nn.Sequential(*layers)

        head_ch = _c(960, w)
        self.head = nn.Sequential(
            ConvBNAct(in_ch, head_ch, kernel_size=1, stride=1, padding=0, act="relu"),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(head_ch, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.features(x)
        return self.head(x)


_VARIANTS: dict[str, tuple[float, float]] = {
    # (wm, se_ratio)
    "ghostnet_0_5": (0.5, 0.25),
    "ghostnet_0_75": (0.75, 0.25),
    "ghostnet_1_0": (1.0, 0.25),
    "ghostnet_1_3": (1.3, 0.25),
    "ghostnet_1_5": (1.5, 0.25),
    "ghostnetv2_1_0": (1.0, 0.25),
    "ghostnetv2_1_3": (1.3, 0.25),
    # aliases
    "ghostnet": (1.0, 0.25),
}


def build_ghostnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "ghostnet_1_0",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            "Unknown GhostNet variant. Supported: ghostnet_0_5|ghostnet_0_75|ghostnet_1_0|ghostnet_1_3|ghostnet_1_5|ghostnetv2_1_0|ghostnetv2_1_3"
        )
    wm, se_ratio = _VARIANTS[name]
    return GhostNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult) * float(wm),
        dropout=float(dropout),
        se_ratio=float(se_ratio),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["ghostnet_0_5", "ghostnet_1_0", "ghostnet_1_3"]:
        m = build_ghostnet_classifier(in_channels=3, num_classes=10, variant=v, width_mult=1.0)
        y = m(x)
        print(v, tuple(y.shape))

