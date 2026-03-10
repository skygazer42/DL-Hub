import torch
from torch import nn

from dlhub.vision.backbones._blocks import (
    ConvBNAct,
    GlobalAvgPoolHead,
    InvertedResidual,
    make_divisible,
)


class FBNetV3Classifier(nn.Module):
    """FBNetV3-style inverted residual network (simplified)."""

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
            ConvBNAct(int(in_channels), c(16), kernel_size=3, stride=2, act="silu"),
            ConvBNAct(c(16), c(16), kernel_size=3, stride=1, groups=c(16), act="silu"),
        )

        blocks: list[nn.Module] = []
        cfg = [
            # in, out, stride, exp, se
            (c(16), c(24), 2, 4.0, None),
            (c(24), c(24), 1, 4.0, None),
            (c(24), c(40), 2, 4.0, 0.25),
            (c(40), c(40), 1, 4.0, 0.25),
            (c(40), c(80), 2, 6.0, 0.25),
            (c(80), c(80), 1, 6.0, 0.25),
            (c(80), c(112), 1, 6.0, 0.25),
            (c(112), c(160), 2, 6.0, 0.25),
        ]
        for in_c, out_c, s, exp, se in cfg:
            blocks.append(
                InvertedResidual(
                    int(in_c),
                    int(out_c),
                    stride=int(s),
                    expand_ratio=float(exp),
                    se_ratio=float(se) if se is not None else None,
                    act="silu",
                    drop_path=0.0,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            ConvBNAct(c(160), c(960), kernel_size=1, stride=1, padding=0, act="silu"),
            GlobalAvgPoolHead(c(960), int(num_classes), dropout=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "fbnet_v3_a": {"w": 0.75},
    "fbnet_v3_b": {"w": 1.0},
    "fbnet_v3_c": {"w": 1.25},
}


def build_fbnet_v3_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "fbnet_v3_b",
    dropout: float = 0.2,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown FBNetV3 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return FBNetV3Classifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(spec["w"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_fbnet_v3_classifier(in_channels=3, num_classes=10, variant="fbnet_v3_a")
    y = m(x)
    print("fbnet_v3_a", tuple(y.shape))
