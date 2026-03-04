from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._transformer import TransformerEncoderBlock, PatchEmbed


class MAEViTClassifier(nn.Module):
    """MAE-style ViT encoder for classification (decoder omitted here)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int = 64,
        patch_size: int = 4,
        dim: int = 256,
        depth: int = 8,
        heads: int = 8,
        dropout: float = 0.1,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch = PatchEmbed(int(in_channels), int(dim), patch_size=int(patch_size))
        h = int(image_size) // int(patch_size)
        w = int(image_size) // int(patch_size)
        self.pos = nn.Parameter(torch.zeros(1, h * w, int(dim)))
        dp_rates = torch.linspace(0.0, float(drop_path), steps=int(depth)).tolist()
        self.blocks = nn.Sequential(
            *[TransformerEncoderBlock(int(dim), int(heads), mlp_ratio=4.0, dropout=0.0, drop_path=float(dp_rates[i])) for i in range(int(depth))]
        )
        self.norm = nn.LayerNorm(int(dim))
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(int(dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch(x)
        x = self.blocks(x + self.pos)
        x = self.norm(x)
        x = self.drop(x.mean(dim=1))
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "maevit_tiny": {"dim": 256, "depth": 8, "heads": 8, "patch": 4},
    "maevit_small": {"dim": 384, "depth": 10, "heads": 12, "patch": 4},
}


def build_maevit_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "maevit_tiny",
    image_size: int = 64,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MAE-ViT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return MAEViTClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        patch_size=int(spec["patch"]),
        dim=int(spec["dim"]),
        depth=int(spec["depth"]),
        heads=int(spec["heads"]),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_maevit_classifier(in_channels=3, num_classes=10, variant="maevit_tiny", image_size=64)
    y = m(x)
    print("maevit_tiny", tuple(y.shape))

