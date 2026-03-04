from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ChannelShuffle, ConvBNAct, GlobalAvgPoolHead, InvertedResidual, make_divisible


class MicroBlock(nn.Module):
    """A tiny MicroNet-inspired block: grouped pointwise + shuffle + depthwise."""

    def __init__(self, in_ch: int, out_ch: int, *, stride: int, groups: int = 2) -> None:
        super().__init__()
        g = int(groups)
        c_in = int(in_ch)
        c_out = int(out_ch)
        self.pw1 = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1, groups=g, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            ChannelShuffle(g),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=int(stride), padding=1, groups=c_out, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        self.pw2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=1, groups=g, bias=False),
            nn.BatchNorm2d(c_out),
        )
        self.act = nn.ReLU(inplace=True)
        self.down: nn.Module | None = None
        if int(stride) != 1 or c_in != c_out:
            self.down = nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=1, stride=int(stride), bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.down is None else self.down(x)
        x = self.pw1(x)
        x = self.dw(x)
        x = self.pw2(x)
        return self.act(x + identity)


class MicroNetClassifier(nn.Module):
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

        self.stem = ConvBNAct(int(in_channels), c(16), kernel_size=3, stride=2, act="relu")
        self.stage1 = nn.Sequential(MicroBlock(c(16), c(32), stride=2, groups=2), MicroBlock(c(32), c(32), stride=1, groups=2))
        self.stage2 = nn.Sequential(
            MicroBlock(c(32), c(64), stride=2, groups=4),
            MicroBlock(c(64), c(64), stride=1, groups=4),
            MicroBlock(c(64), c(64), stride=1, groups=4),
        )
        self.stage3 = nn.Sequential(
            MicroBlock(c(64), c(128), stride=2, groups=4),
            MicroBlock(c(128), c(128), stride=1, groups=4),
        )
        self.head = nn.Sequential(ConvBNAct(c(128), c(512), kernel_size=1, stride=1, padding=0, act="relu"), GlobalAvgPoolHead(c(512), int(num_classes), dropout=float(dropout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "micronet_xs": {"w": 0.5},
    "micronet_s": {"w": 0.75},
    "micronet_m": {"w": 1.0},
}


def build_micronet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "micronet_m",
    dropout: float = 0.2,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MicroNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return MicroNetClassifier(in_channels=int(in_channels), num_classes=int(num_classes), width_mult=float(spec["w"]), dropout=float(dropout))


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_micronet_classifier(in_channels=3, num_classes=10, variant="micronet_xs")
    y = m(x)
    print("micronet_xs", tuple(y.shape))

