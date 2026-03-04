from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, InvertedResidual, SqueezeExcite, scale_channels


class EdgeNeXtBlock(nn.Module):
    """EdgeNeXt-inspired lightweight block (MBConv + optional SE)."""

    def __init__(self, in_ch: int, out_ch: int, *, stride: int, expand: float, se_ratio: float | None) -> None:
        super().__init__()
        self.block = InvertedResidual(
            int(in_ch),
            int(out_ch),
            stride=int(stride),
            expand_ratio=float(expand),
            se_ratio=float(se_ratio) if se_ratio is not None else None,
            act="silu",
            drop_path=0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EdgeNeXtClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        widths: tuple[int, int, int, int] = (32, 64, 128, 256),
        depths: tuple[int, int, int, int] = (2, 2, 4, 2),
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        chs = tuple(scale_channels(int(c), float(width_mult), min_ch=16, divisor=8) for c in widths)
        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), chs[0], kernel_size=3, stride=2, act="silu"),
            ConvBNAct(chs[0], chs[0], kernel_size=3, stride=1, act="silu"),
        )

        def make_stage(in_ch: int, out_ch: int, depth: int, *, stride: int) -> nn.Sequential:
            blocks: list[nn.Module] = []
            blocks.append(EdgeNeXtBlock(in_ch, out_ch, stride=int(stride), expand=4.0, se_ratio=0.25))
            for _ in range(int(depth) - 1):
                blocks.append(EdgeNeXtBlock(out_ch, out_ch, stride=1, expand=4.0, se_ratio=0.25))
            return nn.Sequential(*blocks)

        self.stage1 = make_stage(chs[0], chs[0], depths[0], stride=1)
        self.stage2 = make_stage(chs[0], chs[1], depths[1], stride=2)
        self.stage3 = make_stage(chs[1], chs[2], depths[2], stride=2)
        self.stage4 = make_stage(chs[2], chs[3], depths[3], stride=2)

        self.head = GlobalAvgPoolHead(chs[3], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "edgenext_xxs": {"widths": (24, 48, 96, 192), "depths": (2, 2, 4, 2)},
    "edgenext_xs": {"widths": (32, 64, 128, 256), "depths": (2, 2, 4, 2)},
    "edgenext_s": {"widths": (40, 80, 160, 320), "depths": (2, 2, 6, 2)},
}


def build_edgenext_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "edgenext_xs",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown EdgeNeXt variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return EdgeNeXtClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        widths=tuple(map(int, spec["widths"])),
        depths=tuple(map(int, spec["depths"])),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_edgenext_classifier(in_channels=3, num_classes=10, variant="edgenext_xxs", width_mult=0.5)
    y = m(x)
    print("edgenext_xxs", tuple(y.shape))

