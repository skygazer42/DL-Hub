from __future__ import annotations

import math

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, InvertedResidual, make_divisible


class ReXNetClassifier(nn.Module):
    """ReXNet (linear channel progression) style network (simplified)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        stem_channels: int = 32,
        final_channels: int = 512,
        num_blocks: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        w = float(width_mult)
        d = float(depth_mult)
        n = max(1, int(round(int(num_blocks) * d)))
        c0 = make_divisible(int(round(int(stem_channels) * w)), 8)
        c_final = make_divisible(int(round(int(final_channels) * w)), 8)

        self.stem = ConvBNAct(int(in_channels), c0, kernel_size=3, stride=2, act="silu")

        # Linear channel schedule
        chs: list[int] = []
        for i in range(n + 1):
            t = i / max(1, n)
            chs.append(int(round((1 - t) * c0 + t * c_final)))
        # Ensure divisibility
        chs = [make_divisible(c, 8) for c in chs]

        blocks: list[nn.Module] = []
        in_ch = c0
        for i in range(n):
            out_ch = chs[i + 1]
            stride = 2 if i in {0, n // 3, 2 * n // 3} else 1
            blocks.append(
                InvertedResidual(
                    in_ch,
                    out_ch,
                    stride=int(stride),
                    expand_ratio=6.0,
                    se_ratio=0.25,
                    act="silu",
                    drop_path=0.0,
                )
            )
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            ConvBNAct(in_ch, c_final, kernel_size=1, stride=1, padding=0, act="silu"),
            GlobalAvgPoolHead(c_final, int(num_classes), dropout=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "rexnet_1_0": {"w": 1.0, "d": 1.0},
    "rexnet_1_3": {"w": 1.3, "d": 1.0},
    "rexnet_2_0": {"w": 2.0, "d": 1.0},
    "rexnet_small": {"w": 0.75, "d": 0.75},
}


def build_rexnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "rexnet_1_0",
    dropout: float = 0.2,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ReXNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return ReXNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(spec["w"]),
        depth_mult=float(spec["d"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_rexnet_classifier(in_channels=3, num_classes=10, variant="rexnet_small")
    y = m(x)
    print("rexnet_small", tuple(y.shape))

