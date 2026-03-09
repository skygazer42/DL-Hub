
import math

import torch
from torch import nn

from dlhub.vision.backbones._blocks import GlobalAvgPoolHead, InvertedResidual, make_divisible


class EfficientNetEdgeClassifier(nn.Module):
    """EfficientNet-EdgeTPU style network (simplified).

    Uses MBConv with SiLU and a relatively wide head.
    """

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

        cfg = [
            (c(24), 1, 1.0, d(1), None),
            (c(32), 2, 6.0, d(2), None),
            (c(48), 2, 6.0, d(2), 0.25),
            (c(96), 2, 6.0, d(3), 0.25),
            (c(160), 2, 6.0, d(4), 0.25),
            (c(256), 1, 6.0, d(2), 0.25),
        ]

        blocks: list[nn.Module] = []
        in_ch = c(32)
        for out_ch, stride, exp, reps, se in cfg:
            for i in range(int(reps)):
                s = int(stride) if i == 0 else 1
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

        head_ch = c(1536)
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
    "efficientnet_edge_s": {"w": 1.0, "d": 1.0},
    "efficientnet_edge_m": {"w": 1.2, "d": 1.1},
    "efficientnet_edge_l": {"w": 1.4, "d": 1.2},
}


def build_efficientnet_edge_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "efficientnet_edge_s",
    dropout: float = 0.2,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown EfficientNet-Edge variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return EfficientNetEdgeClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(spec["w"]),
        depth_mult=float(spec["d"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_efficientnet_edge_classifier(in_channels=3, num_classes=10, variant="efficientnet_edge_s")
    y = m(x)
    print("efficientnet_edge_s", tuple(y.shape))

