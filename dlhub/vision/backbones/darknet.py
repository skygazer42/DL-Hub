
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


class DarkNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int,
        stage_channels: tuple[int, int, int, int],
        stage_blocks: tuple[int, int, int, int],
        width_mult: float,
        dropout: float,
    ) -> None:
        super().__init__()
        w = float(width_mult)
        stem = scale_channels(int(stem_channels), w, min_ch=16, divisor=8)
        self.stem = DarkConv(int(in_channels), stem, kernel_size=3, stride=1)

        in_ch = stem
        stages: list[nn.Module] = []
        for out_base, blocks in zip(stage_channels, stage_blocks, strict=True):
            out_ch = scale_channels(int(out_base), w, min_ch=32, divisor=8)
            stage_layers: list[nn.Module] = [DarkConv(in_ch, out_ch, kernel_size=3, stride=2)]
            stage_layers.extend([DarkResidual(out_ch) for _ in range(int(blocks))])
            stages.append(nn.Sequential(*stage_layers))
            in_ch = out_ch
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


def build_darknet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "darknet53",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name in {"darknet19", "dn19"}:
        return DarkNetClassifier(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            stem_channels=32,
            stage_channels=(64, 128, 256, 512),
            stage_blocks=(1, 2, 2, 1),
            width_mult=float(width_mult),
            dropout=float(dropout),
        )
    if name in {"darknet53", "dn53"}:
        return DarkNetClassifier(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            stem_channels=32,
            stage_channels=(64, 128, 256, 512),
            stage_blocks=(1, 2, 8, 4),
            width_mult=float(width_mult),
            dropout=float(dropout),
        )
    if name in {"darknet_tiny", "tiny"}:
        return DarkNetClassifier(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            stem_channels=16,
            stage_channels=(32, 64, 128, 256),
            stage_blocks=(0, 1, 1, 1),
            width_mult=float(width_mult),
            dropout=float(dropout),
        )
    raise ValueError("Unknown DarkNet variant. Supported: darknet19|darknet53|darknet_tiny")


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["darknet_tiny", "darknet19", "darknet53"]:
        m = build_darknet_classifier(in_channels=3, num_classes=10, variant=v, width_mult=0.5)
        y = m(x)
        print(v, tuple(y.shape))

