from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, DepthwiseSeparableConv, GlobalAvgPoolHead, SqueezeExcite, make_divisible


class LCNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int, se: bool) -> None:
        super().__init__()
        self.conv = DepthwiseSeparableConv(int(in_ch), int(out_ch), stride=int(stride), act="hswish")
        self.se = SqueezeExcite(int(out_ch), se_ratio=0.25) if bool(se) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.se(x)


class PPLCNetClassifier(nn.Module):
    """PP-LCNet style network (simplified)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width_mult: float = 1.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        w = float(width_mult)

        def c(ch: int) -> int:
            return make_divisible(int(round(int(ch) * w)), 8)

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), c(16), kernel_size=3, stride=2, act="hswish"),
            DepthwiseSeparableConv(c(16), c(32), stride=1, act="hswish"),
        )

        self.stage1 = nn.Sequential(LCNetBlock(c(32), c(64), stride=2, se=False), LCNetBlock(c(64), c(64), stride=1, se=False))
        self.stage2 = nn.Sequential(
            LCNetBlock(c(64), c(128), stride=2, se=True),
            LCNetBlock(c(128), c(128), stride=1, se=True),
        )
        self.stage3 = nn.Sequential(
            LCNetBlock(c(128), c(256), stride=2, se=True),
            LCNetBlock(c(256), c(256), stride=1, se=True),
            LCNetBlock(c(256), c(256), stride=1, se=True),
        )
        self.stage4 = nn.Sequential(
            LCNetBlock(c(256), c(512), stride=2, se=True),
            LCNetBlock(c(512), c(512), stride=1, se=True),
        )

        self.head = GlobalAvgPoolHead(c(512), int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "pp_lcnet_0_5": {"w": 0.5},
    "pp_lcnet_1_0": {"w": 1.0},
    "pp_lcnet_1_5": {"w": 1.5},
}


def build_pp_lcnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "pp_lcnet_1_0",
    dropout: float = 0.2,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown PP-LCNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return PPLCNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(spec["w"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_pp_lcnet_classifier(in_channels=3, num_classes=10, variant="pp_lcnet_0_5")
    y = m(x)
    print("pp_lcnet_0_5", tuple(y.shape))

