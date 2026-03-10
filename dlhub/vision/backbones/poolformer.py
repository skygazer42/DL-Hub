import torch
from torch import nn

from dlhub.vision.backbones._blocks import LayerNorm2d, scale_channels


def _parse_patch_variant(variant: str, *, default_patch_size: int) -> tuple[str, int]:
    name = str(variant).lower().strip()
    patch_size = int(default_patch_size)

    # Support `poolformer_tiny_p8` style.
    if "_p" in name:
        base, suffix = name.rsplit("_p", 1)
        if suffix.isdigit():
            name = base
            patch_size = int(suffix)
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    return name, patch_size


class PoolingTokenMixer(nn.Module):
    def __init__(self, *, kernel_size: int = 3) -> None:
        super().__init__()
        k = int(kernel_size)
        if k <= 0:
            raise ValueError("kernel_size must be > 0")
        self.pool = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x) - x


class PoolFormerBlock(nn.Module):
    def __init__(self, dim: int, *, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(int(dim))
        self.mixer = PoolingTokenMixer(kernel_size=3)
        self.drop1 = nn.Dropout(p=float(dropout))

        self.norm2 = LayerNorm2d(int(dim))
        hidden = max(8, int(round(int(dim) * float(mlp_ratio))))
        self.mlp = nn.Sequential(
            nn.Conv2d(int(dim), hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Conv2d(hidden, int(dim), kernel_size=1, bias=True),
        )
        self.drop2 = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.mixer(self.norm1(x)))
        x = x + self.drop2(self.mlp(self.norm2(x)))
        return x


class PoolFormerClassifier(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if int(image_size) % int(patch_size) != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if int(depth) <= 0:
            raise ValueError("depth must be > 0")

        self.patch_embed = nn.Conv2d(
            int(in_channels),
            int(embed_dim),
            kernel_size=int(patch_size),
            stride=int(patch_size),
            padding=0,
        )
        self.blocks = nn.Sequential(
            *[
                PoolFormerBlock(int(embed_dim), mlp_ratio=float(mlp_ratio), dropout=float(dropout))
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(int(embed_dim))
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(int(embed_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = x.mean(dim=(2, 3))
        x = self.norm(x)
        x = self.drop(x)
        return self.head(x)


_SPECS: dict[str, tuple[int, int]] = {
    "poolformer_tiny": (64, 6),
    "poolformer_small": (96, 8),
    "poolformer_base": (128, 12),
    # friendly aliases
    "tiny": (64, 6),
    "small": (96, 8),
    "base": (128, 12),
    "poolformer": (64, 6),
}


def build_poolformer_classifier(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    variant: str = "poolformer_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name, patch_size = _parse_patch_variant(variant, default_patch_size=4)
    if name not in _SPECS:
        raise ValueError(
            "Unknown PoolFormer variant. Supported: poolformer_tiny|poolformer_small|poolformer_base (+ _p*)"
        )
    base_dim, depth = _SPECS[name]
    embed_dim = scale_channels(int(base_dim), float(width_mult), min_ch=32, divisor=8)
    return PoolFormerClassifier(
        image_size=int(image_size),
        patch_size=int(patch_size),
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        embed_dim=int(embed_dim),
        depth=int(depth),
        mlp_ratio=4.0,
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["poolformer_tiny", "poolformer_small", "poolformer_base", "poolformer_tiny_p8"]:
        m = build_poolformer_classifier(
            in_channels=3, num_classes=10, image_size=64, variant=v, width_mult=1.0
        )
        y = m(x)
        print(v, tuple(y.shape))
