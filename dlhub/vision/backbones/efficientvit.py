
import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, InvertedResidual, SqueezeExcite, make_divisible


class EfficientViTClassifier(nn.Module):
    """EfficientViT-style hybrid (conv-heavy, simplified)."""

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
            ConvBNAct(int(in_channels), c(32), kernel_size=3, stride=2, act="silu"),
            ConvBNAct(c(32), c(32), kernel_size=3, stride=1, act="silu"),
        )

        blocks: list[nn.Module] = []
        cfg = [
            (c(32), c(64), 2, 4.0, 0.25),
            (c(64), c(128), 2, 4.0, 0.25),
            (c(128), c(256), 2, 6.0, 0.25),
            (c(256), c(256), 1, 6.0, 0.25),
        ]
        in_ch = c(32)
        for out_ch, out2, stride, exp, se in cfg:
            blocks.append(InvertedResidual(in_ch, int(out2), stride=int(stride), expand_ratio=float(exp), se_ratio=float(se), act="silu"))
            in_ch = int(out2)
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            ConvBNAct(in_ch, c(1024), kernel_size=1, stride=1, padding=0, act="silu"),
            GlobalAvgPoolHead(c(1024), int(num_classes), dropout=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "efficientvit_m0": {"w": 0.75},
    "efficientvit_m1": {"w": 1.0},
    "efficientvit_m2": {"w": 1.25},
}


def build_efficientvit_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "efficientvit_m1",
    dropout: float = 0.2,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown EfficientViT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return EfficientViTClassifier(in_channels=int(in_channels), num_classes=int(num_classes), width_mult=float(spec["w"]), dropout=float(dropout))


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_efficientvit_classifier(in_channels=3, num_classes=10, variant="efficientvit_m0")
    y = m(x)
    print("efficientvit_m0", tuple(y.shape))

