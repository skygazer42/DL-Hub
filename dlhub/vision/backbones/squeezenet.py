from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


class Fire(nn.Module):
    def __init__(self, in_ch: int, squeeze: int, expand: int) -> None:
        super().__init__()
        self.squeeze = ConvBNAct(in_ch, int(squeeze), kernel_size=1, stride=1, padding=0, act="relu")
        self.expand1 = ConvBNAct(int(squeeze), int(expand), kernel_size=1, stride=1, padding=0, act="relu")
        self.expand3 = ConvBNAct(int(squeeze), int(expand), kernel_size=3, stride=1, padding=1, act="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze(x)
        return torch.cat([self.expand1(x), self.expand3(x)], dim=1)


class SqueezeNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        variant: str = "1_0",
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        name = str(variant).lower().strip()
        if name not in {"1_0", "1_1"}:
            raise ValueError("SqueezeNet variant must be '1_0' or '1_1'")

        w = float(width_mult)

        def c(ch: int) -> int:
            return scale_channels(int(ch), w, min_ch=8, divisor=8)

        stem_out = c(32)
        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), stem_out, kernel_size=3, stride=1, act="relu"),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        if name == "1_0":
            stage_cfg = [
                (c(16), c(32)),
                (c(16), c(32)),
                (c(32), c(64)),
                (c(32), c(64)),
                (c(48), c(96)),
                (c(48), c(96)),
                (c(64), c(128)),
            ]
        else:
            stage_cfg = [
                (c(16), c(32)),
                (c(16), c(32)),
                (c(32), c(64)),
                (c(32), c(64)),
                (c(48), c(96)),
                (c(64), c(128)),
            ]

        layers: list[nn.Module] = []
        in_ch = stem_out
        for i, (sq, ex) in enumerate(stage_cfg):
            layers.append(Fire(in_ch, squeeze=sq, expand=ex))
            in_ch = 2 * int(ex)
            if i in {1, 3}:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.features = nn.Sequential(*layers)

        self.drop = nn.Dropout(p=float(dropout))
        self.classifier = nn.Conv2d(in_ch, int(num_classes), kernel_size=1, stride=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.features(x)
        x = self.drop(x)
        x = self.classifier(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


def build_squeezenet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "1_0",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return SqueezeNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["1_0", "1_1"]:
        m = build_squeezenet_classifier(in_channels=3, num_classes=10, variant=v, width_mult=1.0)
        y = m(x)
        print(f"squeezenet{v}", tuple(y.shape))

