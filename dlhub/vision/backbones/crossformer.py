from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath, GlobalAvgPoolHead, LayerNorm2d, scale_channels


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        if d % h != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = d
        self.num_heads = h
        self.head_dim = d // h
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        b, nq, d = q.shape
        _, nk, _ = kv.shape
        q = self.q(q).view(b, nq, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,Nq,Dh)
        kv = self.kv(kv).view(b, nk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B,H,Nk,Dh)
        attn = (q @ k.transpose(-2, -1)) * float(self.scale)
        attn = torch.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(b, nq, d)
        return self.proj(y)


class CrossFormerBlock(nn.Module):
    """CrossFormer-like: attend from full tokens to pooled tokens (simplified)."""

    def __init__(self, dim: int, num_heads: int, *, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.xattn = CrossAttention(d, int(num_heads))
        self.dp1 = DropPath(float(drop_path))
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.dp2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor, *, hw: tuple[int, int]) -> torch.Tensor:
        b, n, d = x.shape
        h, w = int(hw[0]), int(hw[1])
        x2d = x.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        pooled = nn.functional.avg_pool2d(x2d, kernel_size=2, stride=2)  # (B,D,H/2,W/2)
        kv = pooled.flatten(2).transpose(1, 2)  # (B, Nk, D)
        x = x + self.dp1(self.xattn(self.norm1(x), self.norm_kv(kv)))
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


class CrossFormerClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int = 64,
        patch_size: int = 4,
        dim: int = 192,
        depth: int = 8,
        num_heads: int = 6,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        h = self.image_size // self.patch_size
        w = self.image_size // self.patch_size
        self.hw = (h, w)
        d = int(dim)
        self.patch = nn.Conv2d(int(in_channels), d, kernel_size=self.patch_size, stride=self.patch_size, bias=True)
        self.pos = nn.Parameter(torch.zeros(1, h * w, d))
        dp_rates = torch.linspace(0.0, float(drop_path), steps=int(depth)).tolist()
        self.blocks = nn.ModuleList([CrossFormerBlock(d, int(num_heads), drop_path=float(dp_rates[i])) for i in range(int(depth))])
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(d, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = self.drop(x + self.pos)
        for b in self.blocks:
            x = b(x, hw=self.hw)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "crossformer_tiny": {"dim": 192, "depth": 8, "heads": 6, "patch": 4},
    "crossformer_small": {"dim": 256, "depth": 10, "heads": 8, "patch": 4},
}


def build_crossformer_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "crossformer_tiny",
    image_size: int = 64,
    dropout: float = 0.1,
    drop_path: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CrossFormer variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return CrossFormerClassifier(
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
    m = build_crossformer_classifier(in_channels=3, num_classes=10, variant="crossformer_tiny", image_size=64)
    y = m(x)
    print("crossformer_tiny", tuple(y.shape))

