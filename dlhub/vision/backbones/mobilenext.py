from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, InvertedResidual, make_divisible


class SandglassBlock(nn.Module):
    """MobileNeXt Sandglass block (approximate).

    Sandglass moves expansion to the end; this implementation is a lightweight
    approximation built from depthwise + pointwise convs.
    """

    def __init__(self, in_ch: int, out_ch: int, *, stride: int, expand_ratio: float = 6.0) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        s = int(stride)
        hidden = int(round(c_out * float(expand_ratio)))
        self.use_res = s == 1 and c_in == c_out

        self.dw1 = ConvBNAct(c_in, c_in, kernel_size=3, stride=1, groups=c_in, act="relu6")
        self.pw_expand = ConvBNAct(c_in, hidden, kernel_size=1, stride=1, padding=0, act="relu6")
        self.dw2 = ConvBNAct(hidden, hidden, kernel_size=3, stride=s, groups=hidden, act="relu6")
        self.pw_proj = nn.Sequential(
            nn.Conv2d(hidden, c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw1(x)
        y = self.pw_expand(y)
        y = self.dw2(y)
        y = self.pw_proj(y)
        return x + y if self.use_res else y


class MobileNeXtClassifier(nn.Module):
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

        self.stem = ConvBNAct(int(in_channels), c(32), kernel_size=3, stride=2, act="relu6")

        blocks: list[nn.Module] = []
        cfg = [
            (c(32), c(32), 1, 2.0),
            (c(32), c(64), 2, 6.0),
            (c(64), c(64), 1, 6.0),
            (c(64), c(128), 2, 6.0),
            (c(128), c(128), 1, 6.0),
            (c(128), c(256), 2, 6.0),
            (c(256), c(256), 1, 6.0),
        ]
        for in_c, out_c, s, exp in cfg:
            blocks.append(SandglassBlock(int(in_c), int(out_c), stride=int(s), expand_ratio=float(exp)))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            ConvBNAct(c(256), c(1024), kernel_size=1, stride=1, padding=0, act="relu6"),
            GlobalAvgPoolHead(c(1024), int(num_classes), dropout=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "mobilenext_s": {"w": 0.75},
    "mobilenext_m": {"w": 1.0},
    "mobilenext_l": {"w": 1.25},
}


def build_mobilenext_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "mobilenext_m",
    dropout: float = 0.2,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MobileNeXt variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return MobileNeXtClassifier(in_channels=int(in_channels), num_classes=int(num_classes), width_mult=float(spec["w"]), dropout=float(dropout))


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_mobilenext_classifier(in_channels=3, num_classes=10, variant="mobilenext_s")
    y = m(x)
    print("mobilenext_s", tuple(y.shape))

