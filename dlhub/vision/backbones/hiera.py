import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels
from dlhub.vision.backbones._transformer import TransformerEncoderBlock


class PatchMerging(nn.Module):
    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.out_dim = int(out_dim)
        self.norm = nn.LayerNorm(4 * self.dim)
        self.proj = nn.Linear(4 * self.dim, self.out_dim)

    def forward(
        self, x: torch.Tensor, *, hw: tuple[int, int]
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        b, n, d = x.shape
        h, w = int(hw[0]), int(hw[1])
        if n != h * w:
            raise ValueError("hw mismatch")
        x = x.view(b, h, w, d)
        if h % 2 != 0 or w % 2 != 0:
            x = x[:, : h - (h % 2), : w - (w % 2)]
            h, w = x.shape[1], x.shape[2]
        x0 = x[:, 0::2, 0::2]
        x1 = x[:, 0::2, 1::2]
        x2 = x[:, 1::2, 0::2]
        x3 = x[:, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=-1).view(b, (h // 2) * (w // 2), 4 * d)
        x = self.norm(x)
        x = self.proj(x)
        return x, (h // 2, w // 2)


class HieraStage(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, *, drop_path: float) -> None:
        super().__init__()
        d = int(dim)
        depth = int(depth)
        dp_rates = torch.linspace(0.0, float(drop_path), steps=depth).tolist()
        self.blocks = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    d, int(num_heads), mlp_ratio=4.0, dropout=0.0, drop_path=float(dp_rates[i])
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class HieraClassifier(nn.Module):
    """Hiera-style hierarchical ViT (simplified)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int = 64,
        patch_size: int = 4,
        dims: tuple[int, int, int, int] = (96, 192, 384, 768),
        depths: tuple[int, int, int, int] = (2, 2, 6, 2),
        heads: tuple[int, int, int, int] = (3, 6, 12, 24),
        width_mult: float = 1.0,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        dims = tuple(scale_channels(int(d), float(width_mult), min_ch=16, divisor=8) for d in dims)
        depths = tuple(int(x) for x in depths)
        heads = tuple(int(h) for h in heads)

        self.patch = nn.Conv2d(
            int(in_channels),
            dims[0],
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.hw = (self.image_size // self.patch_size, self.image_size // self.patch_size)

        self.stage1 = HieraStage(dims[0], depths[0], heads[0], drop_path=float(drop_path))
        self.merge1 = PatchMerging(dims[0], dims[1])
        self.stage2 = HieraStage(dims[1], depths[1], heads[1], drop_path=float(drop_path))
        self.merge2 = PatchMerging(dims[1], dims[2])
        self.stage3 = HieraStage(dims[2], depths[2], heads[2], drop_path=float(drop_path))
        self.merge3 = PatchMerging(dims[2], dims[3])
        self.stage4 = HieraStage(dims[3], depths[3], heads[3], drop_path=float(drop_path))

        self.norm = nn.LayerNorm(dims[-1])
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(dims[-1], int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch(x).flatten(2).transpose(1, 2)
        hw = self.hw
        x = self.stage1(x)
        x, hw = self.merge1(x, hw=hw)
        x = self.stage2(x)
        x, hw = self.merge2(x, hw=hw)
        x = self.stage3(x)
        x, hw = self.merge3(x, hw=hw)
        x = self.stage4(x)
        x = self.norm(x)
        x = self.drop(x.mean(dim=1))
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "hiera_tiny": {"dims": (96, 192, 384, 768), "depths": (2, 2, 6, 2), "heads": (3, 6, 12, 24)},
    "hiera_small": {"dims": (96, 192, 384, 768), "depths": (2, 2, 12, 2), "heads": (3, 6, 12, 24)},
}


def build_hiera_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "hiera_tiny",
    image_size: int = 64,
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Hiera variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return HieraClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        patch_size=4,
        dims=tuple(map(int, spec["dims"])),
        depths=tuple(map(int, spec["depths"])),
        heads=tuple(map(int, spec["heads"])),
        width_mult=float(width_mult),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_hiera_classifier(
        in_channels=3, num_classes=10, variant="hiera_tiny", image_size=64, width_mult=0.5
    )
    y = m(x)
    print("hiera_tiny", tuple(y.shape))
