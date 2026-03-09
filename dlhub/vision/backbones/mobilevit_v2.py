
import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, InvertedResidual, make_divisible
from dlhub.vision.backbones._transformer import TransformerEncoderBlock


class MobileViTBlock(nn.Module):
    """MobileViT block (simplified): local conv -> tokens -> transformer -> fusion."""

    def __init__(self, dim: int, depth: int, num_heads: int) -> None:
        super().__init__()
        d = int(dim)
        self.local = nn.Sequential(
            ConvBNAct(d, d, kernel_size=3, stride=1, act="silu"),
            ConvBNAct(d, d, kernel_size=1, stride=1, padding=0, act="silu"),
        )
        self.blocks = nn.Sequential(*[TransformerEncoderBlock(d, int(num_heads), mlp_ratio=2.0, dropout=0.0, drop_path=0.0) for _ in range(int(depth))])
        self.fuse = ConvBNAct(2 * d, d, kernel_size=1, stride=1, padding=0, act="silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.local(x)
        t = y.flatten(2).transpose(1, 2)  # (B, N, C)
        t = self.blocks(t)
        y2 = t.transpose(1, 2).contiguous().view(b, c, h, w)
        return self.fuse(torch.cat([x, y2], dim=1))


class MobileViTV2Classifier(nn.Module):
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
            InvertedResidual(c(16), c(32), stride=2, expand_ratio=4.0, se_ratio=0.25, act="silu"),
        )
        self.stage1 = nn.Sequential(
            InvertedResidual(c(32), c(48), stride=2, expand_ratio=4.0, se_ratio=0.25, act="silu"),
            MobileViTBlock(c(48), depth=2, num_heads=3),
        )
        self.stage2 = nn.Sequential(
            InvertedResidual(c(48), c(64), stride=2, expand_ratio=4.0, se_ratio=0.25, act="silu"),
            MobileViTBlock(c(64), depth=4, num_heads=4),
        )
        self.stage3 = nn.Sequential(
            InvertedResidual(c(64), c(80), stride=2, expand_ratio=6.0, se_ratio=0.25, act="silu"),
            MobileViTBlock(c(80), depth=3, num_heads=5),
        )
        self.head = nn.Sequential(ConvBNAct(c(80), c(640), kernel_size=1, stride=1, padding=0, act="silu"), GlobalAvgPoolHead(c(640), int(num_classes), dropout=float(dropout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "mobilevit_v2_xs": {"w": 0.75},
    "mobilevit_v2_s": {"w": 1.0},
    "mobilevit_v2_m": {"w": 1.25},
}


def build_mobilevit_v2_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "mobilevit_v2_s",
    dropout: float = 0.2,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MobileViT-v2 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return MobileViTV2Classifier(in_channels=int(in_channels), num_classes=int(num_classes), width_mult=float(spec["w"]), dropout=float(dropout))


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_mobilevit_v2_classifier(in_channels=3, num_classes=10, variant="mobilevit_v2_xs")
    y = m(x)
    print("mobilevit_v2_xs", tuple(y.shape))

