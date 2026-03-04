from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, InvertedResidual, make_divisible
from dlhub.vision.backbones._transformer import TransformerEncoderBlock


class NextViTStage(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, depth: int) -> None:
        super().__init__()
        d = int(dim)
        self.blocks = nn.Sequential(*[TransformerEncoderBlock(d, int(num_heads), mlp_ratio=4.0, dropout=0.0, drop_path=0.0) for _ in range(int(depth))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        t = self.blocks(t)
        return t.transpose(1, 2).contiguous().view(b, c, h, w)


class NextViTClassifier(nn.Module):
    """NextViT-ish: mobile stem + transformer stages (simplified)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width_mult: float = 1.0,
        depths: tuple[int, int, int] = (2, 4, 2),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        w = float(width_mult)

        def c(ch: int) -> int:
            return make_divisible(int(round(int(ch) * w)), 8)

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), c(32), kernel_size=3, stride=2, act="silu"),
            InvertedResidual(c(32), c(64), stride=2, expand_ratio=4.0, se_ratio=0.25, act="silu"),
        )
        self.stage1 = nn.Sequential(InvertedResidual(c(64), c(128), stride=2, expand_ratio=6.0, se_ratio=0.25, act="silu"), NextViTStage(c(128), num_heads=4, depth=int(depths[0])))
        self.stage2 = nn.Sequential(InvertedResidual(c(128), c(192), stride=2, expand_ratio=6.0, se_ratio=0.25, act="silu"), NextViTStage(c(192), num_heads=6, depth=int(depths[1])))
        self.stage3 = nn.Sequential(InvertedResidual(c(192), c(256), stride=2, expand_ratio=6.0, se_ratio=0.25, act="silu"), NextViTStage(c(256), num_heads=8, depth=int(depths[2])))
        self.head = nn.Sequential(ConvBNAct(c(256), c(1024), kernel_size=1, stride=1, padding=0, act="silu"), GlobalAvgPoolHead(c(1024), int(num_classes), dropout=float(dropout)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "nextvit_s": {"w": 0.75, "depths": (2, 3, 2)},
    "nextvit_b": {"w": 1.0, "depths": (2, 4, 2)},
    "nextvit_l": {"w": 1.25, "depths": (3, 6, 3)},
}


def build_nextvit_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "nextvit_b",
    dropout: float = 0.2,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown NextViT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return NextViTClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width_mult=float(spec["w"]),
        depths=tuple(map(int, spec["depths"])),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_nextvit_classifier(in_channels=3, num_classes=10, variant="nextvit_s")
    y = m(x)
    print("nextvit_s", tuple(y.shape))

