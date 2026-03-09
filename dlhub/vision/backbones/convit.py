
import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath
from dlhub.vision.backbones._transformer import MLP, MultiheadSelfAttention, PatchEmbed


class ConViTBlock(nn.Module):
    """ConViT-style gated positional self-attention (simplified).

    Mixes MHSA with a depthwise-conv token mixer via a learnable gate.
    """

    def __init__(self, dim: int, num_heads: int, *, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = nn.LayerNorm(d)
        self.attn = MultiheadSelfAttention(d, int(num_heads), dropout=0.0)
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.dp1 = DropPath(float(drop_path))
        self.norm2 = nn.LayerNorm(d)
        self.mlp = MLP(d, 4 * d, dropout=0.0, act="gelu")
        self.dp2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor, *, hw: tuple[int, int]) -> torch.Tensor:
        b, n, d = x.shape
        h, w = int(hw[0]), int(hw[1])
        # MHSA branch
        y_attn = self.attn(self.norm1(x))
        # Conv branch (local bias)
        x2d = x.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        y_conv = nn.functional.conv2d(x2d, weight=torch.eye(d, device=x.device, dtype=x.dtype).view(d, d, 1, 1), bias=None)
        y_conv = nn.functional.conv2d(y_conv, weight=torch.ones(d, 1, 3, 3, device=x.device, dtype=x.dtype) / 9.0, padding=1, groups=d)
        y_conv = y_conv.permute(0, 2, 3, 1).contiguous().view(b, n, d)
        alpha = torch.clamp(self.gate, 0.0, 1.0)
        x = x + self.dp1(alpha * y_attn + (1.0 - alpha) * y_conv)
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


class ConViTClassifier(nn.Module):
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
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        h = self.image_size // self.patch_size
        w = self.image_size // self.patch_size
        self.hw = (h, w)
        self.patch = PatchEmbed(int(in_channels), int(dim), patch_size=int(patch_size))
        self.pos = nn.Parameter(torch.zeros(1, h * w, int(dim)))
        dp_rates = torch.linspace(0.0, float(drop_path), steps=int(depth)).tolist()
        self.blocks = nn.ModuleList([ConViTBlock(int(dim), int(heads), drop_path=float(dp_rates[i])) for i in range(int(depth))])
        self.norm = nn.LayerNorm(int(dim))
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(int(dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch(x)
        x = self.drop(x + self.pos)
        for b in self.blocks:
            x = b(x, hw=self.hw)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "convit_tiny": {"dim": 192, "depth": 8, "heads": 6, "patch": 4},
    "convit_small": {"dim": 256, "depth": 10, "heads": 8, "patch": 4},
}


def build_convit_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "convit_tiny",
    image_size: int = 64,
    dropout: float = 0.1,
    drop_path: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown ConViT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return ConViTClassifier(
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
    m = build_convit_classifier(in_channels=3, num_classes=10, variant="convit_tiny", image_size=64)
    y = m(x)
    print("convit_tiny", tuple(y.shape))

