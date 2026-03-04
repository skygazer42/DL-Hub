from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct


class _ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(int(in_ch), int(out_ch), kernel_size=3, stride=int(stride), act="relu")
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(out_ch), int(out_ch), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(out_ch)),
        )
        self.act = nn.ReLU(inplace=True)
        self.down: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != int(out_ch):
            self.down = nn.Sequential(
                nn.Conv2d(int(in_ch), int(out_ch), kernel_size=1, stride=int(stride), bias=False),
                nn.BatchNorm2d(int(out_ch)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.down is not None:
            identity = self.down(identity)
        return self.act(x + identity)


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.block = _ResBlock(int(in_ch), int(out_ch), stride=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ResUNet(nn.Module):
    """ResUNet-style encoder/decoder (segmentation head)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        widths: tuple[int, int, int, int] = (32, 64, 128, 256),
    ) -> None:
        super().__init__()
        w1, w2, w3, w4 = (int(x) for x in widths)
        self.enc1 = _ResBlock(int(in_channels), w1, stride=1)
        self.enc2 = _ResBlock(w1, w2, stride=2)
        self.enc3 = _ResBlock(w2, w3, stride=2)
        self.enc4 = _ResBlock(w3, w4, stride=2)

        self.dec3 = _Up(w4 + w3, w3)
        self.dec2 = _Up(w3 + w2, w2)
        self.dec1 = _Up(w2 + w1, w1)

        self.out = nn.Conv2d(w1, int(num_classes), kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x = self.enc4(s3)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.out(x)


_VARIANTS: dict[str, dict] = {
    "resunet_tiny": {"widths": (24, 48, 96, 192)},
    "resunet_base": {"widths": (32, 64, 128, 256)},
    "resunet_large": {"widths": (48, 96, 192, 384)},
}


def build_resunet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "resunet_base",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ResUNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return ResUNet(in_channels=int(in_channels), num_classes=int(num_classes), widths=tuple(map(int, spec["widths"])))


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_resunet_classifier(in_channels=3, num_classes=5, variant="resunet_tiny")
    y = m(x)
    print("resunet_tiny", tuple(y.shape))

