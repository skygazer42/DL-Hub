import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath
from dlhub.vision.backbones._transformer import MLP, PatchEmbed, sinusoidal_positional_embedding


class ReAttention(nn.Module):
    """DeepViT Re-Attention (simplified): mixes attention maps across heads."""

    def __init__(self, dim: int, num_heads: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        if d % h != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = d
        self.num_heads = h
        self.head_dim = d // h
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(d, 3 * d, bias=True)
        self.mix = nn.Conv2d(h, h, kernel_size=1, bias=False)
        self.attn_drop = nn.Dropout(p=float(dropout))
        self.proj = nn.Linear(d, d, bias=True)
        self.proj_drop = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, Dh)

        attn = (q @ k.transpose(-2, -1)) * float(self.scale)  # (B, H, N, N)
        attn = torch.softmax(attn, dim=-1)
        attn = self.mix(attn)  # head mixing
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        y = attn @ v  # (B, H, N, Dh)
        y = y.transpose(1, 2).contiguous().view(b, n, d)
        y = self.proj(y)
        return self.proj_drop(y)


class DeepViTBlock(nn.Module):
    def __init__(
        self, dim: int, heads: int, *, mlp_ratio: float, dropout: float, drop_path: float
    ) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = nn.LayerNorm(d)
        self.attn = ReAttention(d, int(heads), dropout=float(dropout))
        self.dp1 = DropPath(float(drop_path))
        self.norm2 = nn.LayerNorm(d)
        self.mlp = MLP(d, int(round(d * float(mlp_ratio))), dropout=float(dropout), act="gelu")
        self.dp2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.attn(self.norm1(x)))
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


class DeepViTClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        embed_dim: int = 192,
        depth: int = 6,
        heads: int = 3,
        patch_size: int = 4,
        dropout: float = 0.1,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        d = int(embed_dim)
        if d % 2 != 0:
            raise ValueError("embed_dim must be even (for sinusoidal positional embedding)")

        self.embed = PatchEmbed(int(in_channels), d, patch_size=int(patch_size))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))

        dp_rates = torch.linspace(0.0, float(drop_path), steps=max(1, int(depth))).tolist()
        self.blocks = nn.Sequential(
            *[
                DeepViTBlock(
                    d,
                    int(heads),
                    mlp_ratio=4.0,
                    dropout=float(dropout),
                    drop_path=float(dp_rates[i]),
                )
                for i in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(d, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        t = self.embed(x)  # (B, N, D)
        b, n, d = t.shape
        cls = self.cls.expand(b, -1, -1)
        t = torch.cat([cls, t], dim=1)
        pe = sinusoidal_positional_embedding(int(t.shape[1]), d, device=t.device).unsqueeze(0)
        t = t + pe
        t = self.blocks(t)
        t = self.norm(t)
        out = self.drop(t[:, 0])
        return self.head(out)


_VARIANTS: dict[str, dict] = {
    "deepvit_tiny": {"dim": 192, "depth": 4, "heads": 3, "patch": 4},
    "deepvit_small": {"dim": 256, "depth": 6, "heads": 4, "patch": 4},
    "deepvit_base": {"dim": 384, "depth": 8, "heads": 6, "patch": 4},
}


def build_deepvit_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "deepvit_tiny",
    patch_size: int | None = None,
    dropout: float = 0.1,
    drop_path: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DeepViT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return DeepViTClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        embed_dim=int(spec["dim"]),
        depth=int(spec["depth"]),
        heads=int(spec["heads"]),
        patch_size=int(spec["patch"]) if patch_size is None else int(patch_size),
        dropout=float(dropout),
        drop_path=float(drop_path),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_deepvit_classifier(in_channels=3, num_classes=10, variant="deepvit_tiny")
    y = m(x)
    print("deepvit_tiny", tuple(y.shape))
