from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels


_VGG_CFGS: dict[str, list[int | str]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "vgg19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _make_layers(cfg: list[int | str], *, in_channels: int, width_mult: float) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    c_in = int(in_channels)
    last = c_in
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        c_out = scale_channels(int(v), float(width_mult), min_ch=8, divisor=8)
        layers.extend(
            [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            ]
        )
        c_in = c_out
        last = c_out
    return nn.Sequential(*layers), int(last)


class VGGClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        variant: str,
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        name = str(variant).lower().strip()
        if name not in _VGG_CFGS:
            raise ValueError(f"Unknown VGG variant: {variant!r}. Supported: {sorted(_VGG_CFGS)}")
        self.features, out_ch = _make_layers(_VGG_CFGS[name], in_channels=int(in_channels), width_mult=float(width_mult))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(out_ch), int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.features(x)
        return self.head(x)


def build_vgg_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "vgg16",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return VGGClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["vgg11", "vgg16"]:
        m = build_vgg_classifier(in_channels=3, num_classes=10, variant=v, width_mult=0.5)
        y = m(x)
        print(v, tuple(y.shape))
