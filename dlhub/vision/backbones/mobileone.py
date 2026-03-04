from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import SqueezeExcite, scale_channels


def _c(ch: int, width_mult: float) -> int:
    return scale_channels(int(ch), float(width_mult), min_ch=8, divisor=8)


class MobileOneBlock(nn.Module):
    """MobileOne block (RepVGG-ish multi-branch conv)."""

    def __init__(self, in_ch: int, out_ch: int, *, stride: int, num_branches: int, dropout: float) -> None:
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.stride = int(stride)

        branches: list[nn.Module] = []
        for _ in range(int(num_branches)):
            branches.append(
                nn.Sequential(
                    nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, stride=self.stride, padding=1, bias=False),
                    nn.BatchNorm2d(self.out_ch),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, stride=self.stride, padding=0, bias=False),
            nn.BatchNorm2d(self.out_ch),
        )
        self.identity = nn.BatchNorm2d(self.in_ch) if (self.in_ch == self.out_ch and self.stride == 1) else None
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        out = self.branch_1x1(x)
        for br in self.branches:
            out = out + br(x)
        if self.identity is not None:
            out = out + self.identity(x)
        out = self.relu(out)
        return self.drop(out)


class MobileOneClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width_mult: float,
        dropout: float,
        stage_blocks: tuple[int, int, int, int],
        num_branches: int,
        use_se: bool,
    ) -> None:
        super().__init__()
        w = float(width_mult)

        def c(ch: int) -> int:
            return _c(ch, w)

        base = c(32)
        self.stem = MobileOneBlock(
            int(in_channels), base, stride=2, num_branches=int(num_branches), dropout=float(dropout)
        )

        def make_stage(in_ch: int, out_ch: int, blocks: int, first_stride: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            for i in range(int(blocks)):
                stride = int(first_stride) if i == 0 else 1
                layers.append(
                    MobileOneBlock(
                        int(in_ch) if i == 0 else int(out_ch),
                        int(out_ch),
                        stride=stride,
                        num_branches=int(num_branches),
                        dropout=float(dropout),
                    )
                )
                if use_se:
                    layers.append(SqueezeExcite(int(out_ch), se_ratio=0.25))
            return nn.Sequential(*layers)

        self.stage1 = make_stage(base, c(64), blocks=int(stage_blocks[0]), first_stride=1)
        self.stage2 = make_stage(c(64), c(128), blocks=int(stage_blocks[1]), first_stride=2)
        self.stage3 = make_stage(c(128), c(256), blocks=int(stage_blocks[2]), first_stride=2)
        self.stage4 = make_stage(c(256), c(512), blocks=int(stage_blocks[3]), first_stride=2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(c(512), int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


def build_mobileone_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "mobileone_s0",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name in {"s0", "mobileone_s0", "mobileone"}:
        stage_blocks = (1, 1, 3, 1)
        branches = 1
    elif name in {"s1", "mobileone_s1"}:
        stage_blocks = (1, 2, 4, 1)
        branches = 1
    elif name in {"s2", "mobileone_s2"}:
        stage_blocks = (1, 2, 6, 2)
        branches = 2
    elif name in {"s3", "mobileone_s3"}:
        stage_blocks = (2, 3, 8, 2)
        branches = 2
    elif name in {"s4", "mobileone_s4"}:
        stage_blocks = (2, 4, 10, 2)
        branches = 3
    elif name in {"s1_se", "mobileone_s1_se"}:
        stage_blocks = (1, 2, 4, 1)
        branches = 1
    elif name in {"s2_se", "mobileone_s2_se"}:
        stage_blocks = (1, 2, 6, 2)
        branches = 2
    elif name in {"s3_se", "mobileone_s3_se"}:
        stage_blocks = (2, 3, 8, 2)
        branches = 2
    elif name in {"s4_se", "mobileone_s4_se"}:
        stage_blocks = (2, 4, 10, 2)
        branches = 3
    else:
        raise ValueError("Unknown MobileOne variant. Supported: s0|s1|s2|s3|s4 (+ _se variants)")

    use_se = name.endswith("_se")
    return MobileOneClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(width_mult),
        dropout=float(dropout),
        stage_blocks=tuple(map(int, stage_blocks)),
        num_branches=int(branches),
        use_se=bool(use_se),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["mobileone_s0", "mobileone_s2", "mobileone_s2_se"]:
        m = build_mobileone_classifier(in_channels=3, num_classes=10, variant=v, width_mult=1.0)
        y = m(x)
        print(v, tuple(y.shape))

