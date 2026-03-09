
import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, SqueezeExcite, scale_channels


def _c(ch: int, width_mult: float) -> int:
    return scale_channels(int(ch), float(width_mult), min_ch=8, divisor=8)


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        stride: int,
        expand_ratio: int,
        kernel_size: int,
        se_ratio: float,
    ) -> None:
        super().__init__()
        self.use_res = int(stride) == 1 and int(in_ch) == int(out_ch)
        hidden = int(in_ch) * int(expand_ratio)

        layers: list[nn.Module] = []
        if hidden != int(in_ch):
            layers.append(ConvBNAct(in_ch, hidden, kernel_size=1, stride=1, padding=0, act="relu"))
        layers.append(
            ConvBNAct(
                hidden,
                hidden,
                kernel_size=int(kernel_size),
                stride=int(stride),
                groups=hidden,
                act="relu",
            )
        )
        if float(se_ratio) > 0:
            layers.append(SqueezeExcite(hidden, se_ratio=float(se_ratio)))
        layers.append(
            nn.Sequential(
                nn.Conv2d(hidden, int(out_ch), kernel_size=1, bias=False),
                nn.BatchNorm2d(int(out_ch)),
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.use_res:
            out = out + x
        return out


class MNASNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width_mult: float,
        dropout: float,
        variant: str,
    ) -> None:
        super().__init__()
        w = float(width_mult)
        name = str(variant).lower().strip()
        if name in {"0_5", "mnasnet0_5"}:
            wm = 0.5
        elif name in {"0_75", "mnasnet0_75"}:
            wm = 0.75
        elif name in {"1_0", "mnasnet1_0", "mnasnet"}:
            wm = 1.0
        elif name in {"1_3", "mnasnet1_3"}:
            wm = 1.3
        else:
            raise ValueError("Unknown MNASNet variant. Supported: 0_5|0_75|1_0|1_3")

        def c(ch: int) -> int:
            return _c(ch, w * wm)

        self.stem = ConvBNAct(int(in_channels), c(32), kernel_size=3, stride=2, act="relu")

        cfg = [
            # (k, exp, out, n, s, se)
            (3, 1, 16, 1, 1, 0.0),
            (3, 6, 24, 3, 2, 0.0),
            (5, 3, 40, 3, 2, 0.25),
            (3, 6, 80, 3, 2, 0.0),
            (5, 6, 112, 2, 1, 0.25),
            (5, 6, 160, 4, 2, 0.25),
            (3, 6, 320, 1, 1, 0.0),
        ]

        layers: list[nn.Module] = []
        in_ch = c(32)
        for k, t, out, n, s, se in cfg:
            out_ch = c(out)
            for i in range(int(n)):
                stride = int(s) if i == 0 else 1
                layers.append(
                    InvertedResidual(
                        in_ch,
                        out_ch,
                        stride=stride,
                        expand_ratio=int(t),
                        kernel_size=int(k),
                        se_ratio=float(se),
                    )
                )
                in_ch = out_ch
        self.features = nn.Sequential(*layers)

        head_ch = c(1280)
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


def build_mnasnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "mnasnet1_0",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return MNASNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult),
        dropout=float(dropout),
        variant=str(variant),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3"]:
        m = build_mnasnet_classifier(in_channels=3, num_classes=10, variant=v, width_mult=1.0)
        y = m(x)
        print(v, tuple(y.shape))

