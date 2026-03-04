from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath
from dlhub.vision.backbones._transformer import PatchEmbed, TransformerEncoderBlock


class HaarMix(nn.Module):
    """Wavelet-ish mixer using fixed Haar-like filters (stride=1)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        # simple 3x3 edge filters (fixed)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4.0
        sobel_y = sobel_x.t()
        lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]) / 4.0
        bank = torch.stack([sobel_x, sobel_y, lap], dim=0)  # (3,3,3)
        w = bank[:, None, :, :].repeat(c, 1, 1, 1)  # (3C,1,3,3)
        self.register_buffer("w", w)
        self.pw = nn.Conv2d(3 * c, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = nn.functional.conv2d(x, self.w, padding=1, groups=c)
        return self.pw(y)


class WaveViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, *, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        self.mix = HaarMix(d)
        self.dp0 = DropPath(float(drop_path))
        self.attn = TransformerEncoderBlock(d, int(num_heads), mlp_ratio=4.0, dropout=0.0, drop_path=float(drop_path))

    def forward(self, x: torch.Tensor, *, hw: tuple[int, int]) -> torch.Tensor:
        b, n, d = x.shape
        h, w = int(hw[0]), int(hw[1])
        x2d = x.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        x2d = x2d + self.dp0(self.mix(x2d))
        x = x2d.flatten(2).transpose(1, 2)
        x = self.attn(x)
        return x


class WaveViTClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int = 64,
        patch_size: int = 4,
        dim: int = 192,
        depth: int = 8,
        heads: int = 6,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.hw = (int(image_size) // self.patch_size, int(image_size) // self.patch_size)
        self.patch = PatchEmbed(int(in_channels), int(dim), patch_size=int(patch_size))
        self.pos = nn.Parameter(torch.zeros(1, self.hw[0] * self.hw[1], int(dim)))
        dp_rates = torch.linspace(0.0, float(drop_path), steps=int(depth)).tolist()
        self.blocks = nn.ModuleList([WaveViTBlock(int(dim), int(heads), drop_path=float(dp_rates[i])) for i in range(int(depth))])
        self.norm = nn.LayerNorm(int(dim))
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(int(dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch(x) + self.pos
        for b in self.blocks:
            x = b(x, hw=self.hw)
        x = self.norm(x)
        x = self.drop(x.mean(dim=1))
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "wave_vit_tiny": {"dim": 192, "depth": 8, "heads": 6, "patch": 4},
    "wave_vit_small": {"dim": 256, "depth": 10, "heads": 8, "patch": 4},
}


def build_wave_vit_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "wave_vit_tiny",
    image_size: int = 64,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Wave-ViT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return WaveViTClassifier(
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
    x = torch.randn(1, 3, 64, 64)
    m = build_wave_vit_classifier(in_channels=3, num_classes=10, variant="wave_vit_tiny", image_size=64)
    y = m(x)
    print("wave_vit_tiny", tuple(y.shape))

