import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath
from dlhub.vision.backbones._transformer import MLP, MultiheadSelfAttention, PatchEmbed


class StripAttention(nn.Module):
    """Cross-shaped window attention (simplified) using stripe MHSA.

    Applies MHSA along rows and columns separately and sums the results.
    """

    def __init__(self, dim: int, num_heads: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        self.attn_h = MultiheadSelfAttention(d, h, dropout=float(dropout))
        self.attn_v = MultiheadSelfAttention(d, h, dropout=float(dropout))

    def forward(self, x: torch.Tensor, *, hw: tuple[int, int]) -> torch.Tensor:
        b, n, d = x.shape
        h, w = int(hw[0]), int(hw[1])
        if n != h * w:
            raise ValueError(f"Token count mismatch: n={n} vs h*w={h*w}")
        x_hw = x.view(b, h, w, d)
        # horizontal stripes: (B*H, W, D)
        x_h = x_hw.reshape(b * h, w, d)
        y_h = self.attn_h(x_h).reshape(b, h, w, d)
        # vertical stripes: (B*W, H, D)
        x_v = x_hw.permute(0, 2, 1, 3).reshape(b * w, h, d)
        y_v = self.attn_v(x_v).reshape(b, w, h, d).permute(0, 2, 1, 3)
        y = y_h + y_v
        return y.reshape(b, n, d)


class CSWinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = nn.LayerNorm(d)
        self.attn = StripAttention(d, int(num_heads), dropout=float(dropout))
        self.dp1 = DropPath(float(drop_path))
        self.norm2 = nn.LayerNorm(d)
        self.mlp = MLP(d, int(round(d * float(mlp_ratio))), dropout=float(dropout), act="gelu")
        self.dp2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor, *, hw: tuple[int, int]) -> torch.Tensor:
        x = x + self.dp1(self.attn(self.norm1(x), hw=hw))
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


class CSWinClassifier(nn.Module):
    """CSWin-style ViT (single-stage, fixed resolution, simplified)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int = 64,
        patch_size: int = 4,
        dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        dropout: float = 0.0,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.dim = int(dim)
        h = self.image_size // self.patch_size
        w = self.image_size // self.patch_size
        if h <= 0 or w <= 0:
            raise ValueError("image_size must be >= patch_size")
        self.hw = (h, w)

        self.patch = PatchEmbed(int(in_channels), int(dim), patch_size=int(patch_size))
        self.pos = nn.Parameter(torch.zeros(1, h * w, int(dim)))
        self.drop = nn.Dropout(p=float(dropout))

        dp_rates = torch.linspace(0.0, float(drop_path), steps=int(depth)).tolist()
        self.blocks = nn.ModuleList(
            [
                CSWinBlock(
                    int(dim), int(num_heads), dropout=float(dropout), drop_path=float(dp_rates[i])
                )
                for i in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(int(dim))
        self.head = nn.Linear(int(dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch(x)  # (B, N, D)
        x = x + self.pos
        x = self.drop(x)
        for b in self.blocks:
            x = b(x, hw=self.hw)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "cswin_tiny": {"dim": 192, "depth": 6, "heads": 6, "patch": 4},
    "cswin_small": {"dim": 256, "depth": 8, "heads": 8, "patch": 4},
}


def build_cswin_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "cswin_tiny",
    image_size: int = 64,
    dropout: float = 0.0,
    drop_path: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CSWin variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return CSWinClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        patch_size=int(spec["patch"]),
        dim=int(spec["dim"]),
        depth=int(spec["depth"]),
        num_heads=int(spec["heads"]),
        dropout=float(dropout),
        drop_path=float(drop_path),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_cswin_classifier(in_channels=3, num_classes=10, variant="cswin_tiny", image_size=64)
    y = m(x)
    print("cswin_tiny", tuple(y.shape))
