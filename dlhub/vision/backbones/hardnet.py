from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import (
    CBAM,
    CoordAttention,
    ConvBNAct,
    ECALayer,
    GlobalAvgPoolHead,
    GlobalContextBlock,
    NonLocal2D,
    SKConv,
    SqueezeExcite,
    scale_channels,
)
class _ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int, attn: nn.Module) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=int(stride), padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch))
        self.attn = attn
        self.act = nn.ReLU(inplace=True)

        self.down: nn.Module | None = None
        if int(stride) != 1 or int(in_ch) != int(out_ch):
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=int(stride), bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attn(x)
        if self.down is not None:
            identity = self.down(identity)
        x = x + identity
        return self.act(x)


_VARIANTS: dict[str, dict] = {
    'tiny':  {'channels': (32, 64, 128, 256), 'depths': (1, 1, 2, 1)},
    'small': {'channels': (48, 96, 192, 384), 'depths': (2, 2, 3, 2)},
    'base':  {'channels': (64, 128, 256, 512), 'depths': (2, 3, 4, 2)},
}


class HardnetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        variant: str = 'tiny',
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        name = str(variant).lower().strip()
        if name not in _VARIANTS:
            raise ValueError(f'Unknown variant: {variant!r}. Supported: {sorted(_VARIANTS)}')

        spec = _VARIANTS[name]
        chans = tuple(scale_channels(int(c), float(width_mult), min_ch=16, divisor=8) for c in spec['channels'])
        depths = tuple(int(d) for d in spec['depths'])

        self.stem = ConvBNAct(int(in_channels), chans[0], kernel_size=3, stride=2, act='relu')

        def make_stage(in_ch: int, out_ch: int, depth: int, *, stride: int) -> nn.Sequential:
            blocks: list[nn.Module] = []
            attn_mod = nn.Identity()
            blocks.append(_ResidualBlock(in_ch, out_ch, stride=int(stride), attn=attn_mod))
            for _ in range(int(depth) - 1):
                blocks.append(_ResidualBlock(out_ch, out_ch, stride=1, attn=nn.Identity()))
            return nn.Sequential(*blocks)

        self.stage1 = make_stage(chans[0], chans[0], depths[0], stride=1)
        self.stage2 = make_stage(chans[0], chans[1], depths[1], stride=2)
        self.stage3 = make_stage(chans[1], chans[2], depths[2], stride=2)
        self.stage4 = make_stage(chans[2], chans[3], depths[3], stride=2)

        self.head = GlobalAvgPoolHead(chans[3], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


def build_hardnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = 'tiny',
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return HardnetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_hardnet_classifier(in_channels=3, num_classes=10, variant='tiny', width_mult=1.0)
    y = m(x)
    print('hardnet', tuple(y.shape))
