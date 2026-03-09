
import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, InvertedResidual, make_divisible
from dlhub.vision.backbones._transformer import TransformerEncoderBlock


class EdgeViTBlock(nn.Module):
    """EdgeViT-like block (conv stem + late MHSA, simplified)."""

    def __init__(self, dim: int, *, num_heads: int) -> None:
        super().__init__()
        d = int(dim)
        self.attn = TransformerEncoderBlock(d, int(num_heads), mlp_ratio=4.0, dropout=0.0, drop_path=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        t = self.attn(t)
        return t.transpose(1, 2).contiguous().view(b, c, h, w)


class EdgeViTClassifier(nn.Module):
    def __init__(self, *, in_channels: int, num_classes: int, width_mult: float = 1.0, dropout: float = 0.2) -> None:
        super().__init__()
        w = float(width_mult)

        def c(ch: int) -> int:
            return make_divisible(int(round(int(ch) * w)), 8)

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), c(32), kernel_size=3, stride=2, act="silu"),
            InvertedResidual(c(32), c(64), stride=2, expand_ratio=4.0, se_ratio=0.25, act="silu"),
        )
        self.stage = nn.Sequential(
            InvertedResidual(c(64), c(96), stride=2, expand_ratio=4.0, se_ratio=0.25, act="silu"),
            EdgeViTBlock(c(96), num_heads=4),
            InvertedResidual(c(96), c(160), stride=2, expand_ratio=6.0, se_ratio=0.25, act="silu"),
            EdgeViTBlock(c(160), num_heads=5),
        )
        self.head = nn.Sequential(ConvBNAct(c(160), c(640), kernel_size=1, stride=1, padding=0, act="silu"), GlobalAvgPoolHead(c(640), int(num_classes), dropout=float(dropout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "edgevit_xs": {"w": 0.75},
    "edgevit_s": {"w": 1.0},
    "edgevit_m": {"w": 1.25},
}


def build_edgevit_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "edgevit_s",
    dropout: float = 0.2,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown EdgeViT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return EdgeViTClassifier(in_channels=int(in_channels), num_classes=int(num_classes), width_mult=float(spec["w"]), dropout=float(dropout))


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_edgevit_classifier(in_channels=3, num_classes=10, variant="edgevit_xs")
    y = m(x)
    print("edgevit_xs", tuple(y.shape))

