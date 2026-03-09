
import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath
from dlhub.vision.backbones._transformer import MultiheadSelfAttention, PatchEmbed


class LocalMLP(nn.Module):
    """ViT MLP with an inserted depthwise conv (LocalViT-style, simplified)."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        d = int(dim)
        h = int(hidden)
        self.fc1 = nn.Linear(d, h)
        self.act = nn.GELU()
        self.dw = nn.Conv2d(h, h, kernel_size=3, padding=1, groups=h, bias=False)
        self.fc2 = nn.Linear(h, d)

    def forward(self, x: torch.Tensor, *, hw: tuple[int, int]) -> torch.Tensor:
        b, n, d = x.shape
        h, w = int(hw[0]), int(hw[1])
        x = self.fc1(x)
        x = self.act(x)
        # depthwise conv on tokens
        x2d = x.transpose(1, 2).contiguous().view(b, -1, h, w)
        x2d = self.dw(x2d)
        x = x2d.flatten(2).transpose(1, 2)
        x = self.fc2(x)
        return x


class LocalViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, *, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = nn.LayerNorm(d)
        self.attn = MultiheadSelfAttention(d, int(num_heads), dropout=0.0)
        self.dp1 = DropPath(float(drop_path))
        self.norm2 = nn.LayerNorm(d)
        self.mlp = LocalMLP(d, 4 * d)
        self.dp2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor, *, hw: tuple[int, int]) -> torch.Tensor:
        x = x + self.dp1(self.attn(self.norm1(x)))
        x = x + self.dp2(self.mlp(self.norm2(x), hw=hw))
        return x


class LocalViTClassifier(nn.Module):
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
        self.patch = PatchEmbed(int(in_channels), int(dim), patch_size=int(patch_size))
        h = int(image_size) // int(patch_size)
        w = int(image_size) // int(patch_size)
        self.hw = (h, w)
        self.pos = nn.Parameter(torch.zeros(1, h * w, int(dim)))
        dp_rates = torch.linspace(0.0, float(drop_path), steps=int(depth)).tolist()
        self.blocks = nn.ModuleList([LocalViTBlock(int(dim), int(heads), drop_path=float(dp_rates[i])) for i in range(int(depth))])
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
    "localvit_tiny": {"dim": 192, "depth": 8, "heads": 6, "patch": 4},
    "localvit_small": {"dim": 256, "depth": 10, "heads": 8, "patch": 4},
}


def build_localvit_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "localvit_tiny",
    image_size: int = 64,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown LocalViT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return LocalViTClassifier(
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
    m = build_localvit_classifier(in_channels=3, num_classes=10, variant="localvit_tiny", image_size=64)
    y = m(x)
    print("localvit_tiny", tuple(y.shape))

