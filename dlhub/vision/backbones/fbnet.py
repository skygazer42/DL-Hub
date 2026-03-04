from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import GlobalAvgPoolHead, InvertedResidual, scale_channels


_VARIANTS: dict[str, dict] = {
    'tiny':  {'channels': (16, 24, 40, 80), 'depths': (1, 2, 2, 2)},
    'small': {'channels': (16, 32, 64, 112), 'depths': (2, 3, 3, 2)},
    'base':  {'channels': (24, 40, 80, 160), 'depths': (2, 3, 4, 3)},
}


class FbnetClassifier(nn.Module):
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
        chans = tuple(scale_channels(int(c), float(width_mult), min_ch=8, divisor=8) for c in spec['channels'])
        depths = tuple(int(d) for d in spec['depths'])

        self.stem = nn.Sequential(
            nn.Conv2d(int(in_channels), chans[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(chans[0]),
            nn.ReLU6(inplace=True),
        )

        def stage(in_ch: int, out_ch: int, depth: int, *, stride: int) -> nn.Sequential:
            blocks: list[nn.Module] = []
            blocks.append(InvertedResidual(in_ch, out_ch, stride=int(stride), expand_ratio=6.0, se_ratio=None, act='relu6'))
            for _ in range(int(depth) - 1):
                blocks.append(InvertedResidual(out_ch, out_ch, stride=1, expand_ratio=6.0, se_ratio=None, act='relu6'))
            return nn.Sequential(*blocks)

        self.stage1 = stage(chans[0], chans[0], depths[0], stride=1)
        self.stage2 = stage(chans[0], chans[1], depths[1], stride=2)
        self.stage3 = stage(chans[1], chans[2], depths[2], stride=2)
        self.stage4 = stage(chans[2], chans[3], depths[3], stride=2)

        self.head = GlobalAvgPoolHead(chans[3], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


def build_fbnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = 'tiny',
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return FbnetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_fbnet_classifier(in_channels=3, num_classes=10, variant='tiny')
    y = m(x)
    print('fbnet', tuple(y.shape))
