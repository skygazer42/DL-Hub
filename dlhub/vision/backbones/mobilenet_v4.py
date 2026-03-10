import math

import torch
from torch import nn

from dlhub.vision.backbones._blocks import (
    GlobalAvgPoolHead,
    InvertedResidual,
    SqueezeExcite,
    make_divisible,
)


class FusedMBConv(nn.Module):
    """Fused MBConv (expand + depthwise fused into one conv), simplified."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        stride: int,
        expand_ratio: float = 4.0,
        se_ratio: float | None = None,
        act: str = "silu",
    ) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        s = int(stride)
        hidden = int(round(c_in * float(expand_ratio)))
        self.use_res = s == 1 and c_in == c_out
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, hidden, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True) if act == "silu" else nn.ReLU(inplace=True),
        )
        self.se = (
            SqueezeExcite(hidden, se_ratio=float(se_ratio))
            if se_ratio is not None and float(se_ratio) > 0
            else nn.Identity()
        )
        self.proj = nn.Sequential(
            nn.Conv2d(hidden, c_out, kernel_size=1, bias=False), nn.BatchNorm2d(c_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = self.se(y)
        y = self.proj(y)
        return x + y if self.use_res else y


class MobileNetV4Classifier(nn.Module):
    """A MobileNetV4-like family (fused/MBConv mix, simplified)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        def c(ch: int) -> int:
            return make_divisible(int(round(int(ch) * float(width_mult))), 8)

        def d(n: int) -> int:
            return max(1, int(math.ceil(int(n) * float(depth_mult))))

        self.stem = nn.Sequential(
            nn.Conv2d(int(in_channels), c(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(32)),
            nn.SiLU(inplace=True),
        )

        blocks: list[nn.Module] = []
        # A small set of fused then MBConv blocks
        cfg = [
            ("fused", c(32), c(32), 1, 2.0, None, d(1)),
            ("fused", c(32), c(48), 2, 4.0, None, d(2)),
            ("mbconv", c(48), c(80), 2, 6.0, 0.25, d(3)),
            ("mbconv", c(80), c(160), 2, 6.0, 0.25, d(4)),
        ]
        in_ch = c(32)
        for kind, _, out_ch, stride, exp, se, reps in cfg:
            for i in range(int(reps)):
                s = int(stride) if i == 0 else 1
                if kind == "fused":
                    blocks.append(
                        FusedMBConv(
                            in_ch,
                            int(out_ch),
                            stride=int(s),
                            expand_ratio=float(exp),
                            se_ratio=se,
                            act="silu",
                        )
                    )
                else:
                    blocks.append(
                        InvertedResidual(
                            in_ch,
                            int(out_ch),
                            stride=int(s),
                            expand_ratio=float(exp),
                            se_ratio=float(se) if se is not None else None,
                            act="silu",
                            drop_path=0.0,
                        )
                    )
                in_ch = int(out_ch)
        self.blocks = nn.Sequential(*blocks)

        head_ch = c(1024)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_ch),
            nn.SiLU(inplace=True),
            GlobalAvgPoolHead(head_ch, int(num_classes), dropout=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "mobilenet_v4_small": {"w": 0.75, "d": 0.8},
    "mobilenet_v4_medium": {"w": 1.0, "d": 1.0},
    "mobilenet_v4_large": {"w": 1.25, "d": 1.2},
}


def build_mobilenet_v4_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "mobilenet_v4_medium",
    dropout: float = 0.2,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown MobileNetV4 variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    return MobileNetV4Classifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(spec["w"]),
        depth_mult=float(spec["d"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_mobilenet_v4_classifier(in_channels=3, num_classes=10, variant="mobilenet_v4_small")
    y = m(x)
    print("mobilenet_v4_small", tuple(y.shape))
