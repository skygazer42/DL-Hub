from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath
from dlhub.vision.backbones._transformer import TransformerEncoderBlock, PatchEmbed


class TNTClassifier(nn.Module):
    """Transformer-in-Transformer (TNT) simplified.

    Inner transformer runs on 2x2 sub-patch tokens; outer transformer runs on patch tokens.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int = 64,
        patch_size: int = 8,
        inner_patch: int = 4,
        outer_dim: int = 192,
        inner_dim: int = 96,
        outer_depth: int = 6,
        inner_depth: int = 2,
        heads: int = 6,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.inner_patch = int(inner_patch)
        if self.patch_size % self.inner_patch != 0:
            raise ValueError("patch_size must be divisible by inner_patch")

        # outer patch tokens
        self.outer = PatchEmbed(int(in_channels), int(outer_dim), patch_size=int(patch_size))
        h_out = self.image_size // self.patch_size
        w_out = self.image_size // self.patch_size
        self.hw_out = (h_out, w_out)
        self.pos_out = nn.Parameter(torch.zeros(1, h_out * w_out, int(outer_dim)))

        # inner sub-patch tokens
        self.inner = PatchEmbed(int(in_channels), int(inner_dim), patch_size=int(inner_patch))
        h_in = self.image_size // self.inner_patch
        w_in = self.image_size // self.inner_patch
        self.hw_in = (h_in, w_in)

        # inner and outer transformers
        self.inner_blocks = nn.Sequential(
            *[TransformerEncoderBlock(int(inner_dim), max(1, int(heads) // 2), mlp_ratio=2.0, dropout=0.0, drop_path=0.0) for _ in range(int(inner_depth))]
        )
        dp_rates = torch.linspace(0.0, float(drop_path), steps=int(outer_depth)).tolist()
        self.outer_blocks = nn.Sequential(
            *[TransformerEncoderBlock(int(outer_dim), int(heads), mlp_ratio=4.0, dropout=0.0, drop_path=float(dp_rates[i])) for i in range(int(outer_depth))]
        )

        self.proj_in_to_out = nn.Linear(int(inner_dim), int(outer_dim))
        self.norm = nn.LayerNorm(int(outer_dim))
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(int(outer_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        b = x.shape[0]

        # Inner tokens: (B, N_in, D_in)
        inner = self.inner(x)
        h_in, w_in = self.hw_in
        inner = inner.view(b, h_in, w_in, -1)

        # Group 2x2 inner tokens per outer patch: (B, H_out, 2, W_out, 2, D_in) -> (B*N_out, 4, D_in)
        h_out, w_out = self.hw_out
        inner = inner.view(b, h_out, self.patch_size // self.inner_patch, w_out, self.patch_size // self.inner_patch, -1)
        inner = inner.permute(0, 1, 3, 2, 4, 5).contiguous().view(b * (h_out * w_out), -1, inner.shape[-1])
        inner = self.inner_blocks(inner)
        inner_summary = inner.mean(dim=1).view(b, h_out * w_out, -1)

        # Outer tokens
        outer = self.outer(x)
        outer = outer + self.pos_out + self.proj_in_to_out(inner_summary)
        outer = self.outer_blocks(outer)
        outer = self.norm(outer)
        outer = self.drop(outer.mean(dim=1))
        return self.head(outer)


_VARIANTS: dict[str, dict] = {
    "tnt_tiny": {"outer": 192, "inner": 96, "od": 6, "id": 2, "heads": 6, "patch": 8},
    "tnt_small": {"outer": 256, "inner": 128, "od": 8, "id": 2, "heads": 8, "patch": 8},
}


def build_tnt_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "tnt_tiny",
    image_size: int = 64,
    dropout: float = 0.1,
    drop_path: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown TNT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return TNTClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        patch_size=int(spec["patch"]),
        inner_patch=4,
        outer_dim=int(spec["outer"]),
        inner_dim=int(spec["inner"]),
        outer_depth=int(spec["od"]),
        inner_depth=int(spec["id"]),
        heads=int(spec["heads"]),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_tnt_classifier(in_channels=3, num_classes=10, variant="tnt_tiny", image_size=64)
    y = m(x)
    print("tnt_tiny", tuple(y.shape))

