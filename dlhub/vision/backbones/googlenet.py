from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


class InceptionModule(nn.Module):
    def __init__(
        self,
        in_ch: int,
        *,
        ch1x1: int,
        ch3x3_reduce: int,
        ch3x3: int,
        ch5x5_reduce: int,
        ch5x5: int,
        pool_proj: int,
    ) -> None:
        super().__init__()
        self.b1 = ConvBNAct(in_ch, ch1x1, kernel_size=1, stride=1, padding=0, act="relu")

        self.b2 = nn.Sequential(
            ConvBNAct(in_ch, ch3x3_reduce, kernel_size=1, stride=1, padding=0, act="relu"),
            ConvBNAct(ch3x3_reduce, ch3x3, kernel_size=3, stride=1, padding=1, act="relu"),
        )

        # Use two 3x3 instead of a 5x5 to keep it stable on CPU.
        self.b3 = nn.Sequential(
            ConvBNAct(in_ch, ch5x5_reduce, kernel_size=1, stride=1, padding=0, act="relu"),
            ConvBNAct(ch5x5_reduce, ch5x5, kernel_size=3, stride=1, padding=1, act="relu"),
            ConvBNAct(ch5x5, ch5x5, kernel_size=3, stride=1, padding=1, act="relu"),
        )

        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNAct(in_ch, pool_proj, kernel_size=1, stride=1, padding=0, act="relu"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class GoogLeNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        variant: str = "googlenet",
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        name = str(variant).lower().strip()
        if name not in {"googlenet", "googlenet_tiny"}:
            raise ValueError("Unknown GoogLeNet variant. Supported: googlenet, googlenet_tiny")

        w = float(width_mult) * (0.75 if name == "googlenet_tiny" else 1.0)

        def c(ch: int) -> int:
            return scale_channels(int(ch), w, min_ch=16, divisor=8)

        stem1 = c(64)
        stem2 = c(64)
        stem3 = c(128)

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), stem1, kernel_size=7, stride=2, padding=3, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBNAct(stem1, stem2, kernel_size=1, stride=1, padding=0, act="relu"),
            ConvBNAct(stem2, stem3, kernel_size=3, stride=1, padding=1, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Inception stack (classic-ish channel sizes, scaled by width_mult).
        self.inception3a = InceptionModule(stem3, ch1x1=c(64), ch3x3_reduce=c(96), ch3x3=c(128), ch5x5_reduce=c(16), ch5x5=c(32), pool_proj=c(32))
        c3a = c(64) + c(128) + c(32) + c(32)
        self.inception3b = InceptionModule(c3a, ch1x1=c(128), ch3x3_reduce=c(128), ch3x3=c(192), ch5x5_reduce=c(32), ch5x5=c(96), pool_proj=c(64))
        c3b = c(128) + c(192) + c(96) + c(64)

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule(c3b, ch1x1=c(192), ch3x3_reduce=c(96), ch3x3=c(208), ch5x5_reduce=c(16), ch5x5=c(48), pool_proj=c(64))
        c4a = c(192) + c(208) + c(48) + c(64)
        self.inception4b = InceptionModule(c4a, ch1x1=c(160), ch3x3_reduce=c(112), ch3x3=c(224), ch5x5_reduce=c(24), ch5x5=c(64), pool_proj=c(64))
        c4b = c(160) + c(224) + c(64) + c(64)
        self.inception4c = InceptionModule(c4b, ch1x1=c(128), ch3x3_reduce=c(128), ch3x3=c(256), ch5x5_reduce=c(24), ch5x5=c(64), pool_proj=c(64))
        c4c = c(128) + c(256) + c(64) + c(64)
        self.inception4d = InceptionModule(c4c, ch1x1=c(112), ch3x3_reduce=c(144), ch3x3=c(288), ch5x5_reduce=c(32), ch5x5=c(64), pool_proj=c(64))
        c4d = c(112) + c(288) + c(64) + c(64)
        self.inception4e = InceptionModule(c4d, ch1x1=c(256), ch3x3_reduce=c(160), ch3x3=c(320), ch5x5_reduce=c(32), ch5x5=c(128), pool_proj=c(128))
        c4e = c(256) + c(320) + c(128) + c(128)

        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionModule(c4e, ch1x1=c(256), ch3x3_reduce=c(160), ch3x3=c(320), ch5x5_reduce=c(32), ch5x5=c(128), pool_proj=c(128))
        c5a = c(256) + c(320) + c(128) + c(128)
        self.inception5b = InceptionModule(c5a, ch1x1=c(384), ch3x3_reduce=c(192), ch3x3=c(384), ch5x5_reduce=c(48), ch5x5=c(128), pool_proj=c(128))
        c5b = c(384) + c(384) + c(128) + c(128)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(c5b, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        return self.head(x)


def build_googlenet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "googlenet",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return GoogLeNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["googlenet", "googlenet_tiny"]:
        m = build_googlenet_classifier(in_channels=3, num_classes=10, variant=v, width_mult=1.0)
        y = m(x)
        print(v, tuple(y.shape))

