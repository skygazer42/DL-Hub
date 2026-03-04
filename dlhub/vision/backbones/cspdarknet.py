from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


class DarkConv(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int, stride: int) -> None:
        super().__init__(ConvBNAct(in_ch, out_ch, kernel_size=int(kernel_size), stride=int(stride), act="leaky"))


class DarkResidual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        hidden = max(8, c // 2)
        self.net = nn.Sequential(
            DarkConv(c, hidden, kernel_size=1, stride=1),
            DarkConv(hidden, c, kernel_size=3, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CSPStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, num_blocks: int, width_mult: float) -> None:
        super().__init__()
        w = float(width_mult)
        out_ch = scale_channels(int(out_ch), w, min_ch=32, divisor=8)
        hidden = max(8, out_ch // 2)

        self.down = DarkConv(int(in_ch), out_ch, kernel_size=3, stride=2)
        self.split1 = DarkConv(out_ch, hidden, kernel_size=1, stride=1)
        self.split2 = DarkConv(out_ch, hidden, kernel_size=1, stride=1)
        self.blocks = nn.Sequential(*[DarkResidual(hidden) for _ in range(int(num_blocks))])
        self.merge = DarkConv(hidden * 2, out_ch, kernel_size=1, stride=1)
        self.out_channels = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        y1 = self.blocks(self.split1(x))
        y2 = self.split2(x)
        out = torch.cat([y1, y2], dim=1)
        return self.merge(out)


class CSPDarkNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stage_channels: tuple[int, int, int, int],
        stage_blocks: tuple[int, int, int, int],
        width_mult: float,
        dropout: float,
    ) -> None:
        super().__init__()
        w = float(width_mult)
        stem = scale_channels(32, w, min_ch=16, divisor=8)
        self.stem = nn.Sequential(
            DarkConv(int(in_channels), stem, kernel_size=3, stride=1),
            DarkConv(stem, stem * 2, kernel_size=3, stride=2),
        )

        in_ch = stem * 2
        stages: list[nn.Module] = []
        for out_ch, blocks in zip(stage_channels, stage_blocks, strict=True):
            stage = CSPStage(in_ch, int(out_ch), num_blocks=int(blocks), width_mult=w)
            stages.append(stage)
            in_ch = int(stage.out_channels)
        self.stages = nn.Sequential(*stages)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(in_ch, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)


def build_cspdarknet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "cspdarknet53",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name in {"cspdarknet53", "csp53"}:
        stage_channels = (64, 128, 256, 512)
        stage_blocks = (1, 2, 8, 4)
    elif name in {"cspdarknet_small", "small"}:
        stage_channels = (64, 128, 256, 512)
        stage_blocks = (1, 2, 4, 2)
    elif name in {"cspdarknet_tiny", "tiny"}:
        stage_channels = (32, 64, 128, 256)
        stage_blocks = (1, 1, 2, 1)
    else:
        raise ValueError("Unknown CSPDarkNet variant. Supported: cspdarknet53|cspdarknet_small|cspdarknet_tiny")

    return CSPDarkNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stage_channels=tuple(map(int, stage_channels)),
        stage_blocks=tuple(map(int, stage_blocks)),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["cspdarknet_tiny", "cspdarknet_small", "cspdarknet53"]:
        m = build_cspdarknet_classifier(in_channels=3, num_classes=10, variant=v, width_mult=0.5)
        y = m(x)
        print(v, tuple(y.shape))

