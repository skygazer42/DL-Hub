from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath
from dlhub.vision.backbones._transformer import MLP, PatchEmbed


class FastFormerAttention(nn.Module):
    """FastFormer attention approximation (simplified)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        d = int(dim)
        self.qkv = nn.Linear(d, 3 * d, bias=True)
        self.q_weight = nn.Linear(d, 1, bias=False)
        self.k_weight = nn.Linear(d, 1, bias=False)
        self.proj = nn.Linear(d, d, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # global query
        wq = torch.softmax(self.q_weight(q).squeeze(-1), dim=1).unsqueeze(-1)
        qg = torch.sum(wq * q, dim=1, keepdim=True)
        # global key
        wk = torch.softmax(self.k_weight(k).squeeze(-1), dim=1).unsqueeze(-1)
        kg = torch.sum(wk * k, dim=1, keepdim=True)
        attn = torch.sigmoid(q * kg)  # (B,N,D)
        y = attn * v + qg * v
        return self.proj(y)


class FastFormerBlock(nn.Module):
    def __init__(self, dim: int, *, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = nn.LayerNorm(d)
        self.attn = FastFormerAttention(d)
        self.dp1 = DropPath(float(drop_path))
        self.norm2 = nn.LayerNorm(d)
        self.mlp = MLP(d, 4 * d, dropout=0.0, act="gelu")
        self.dp2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.attn(self.norm1(x)))
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


class FastFormerViTClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int = 64,
        patch_size: int = 4,
        dim: int = 192,
        depth: int = 10,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch = PatchEmbed(int(in_channels), int(dim), patch_size=int(patch_size))
        h = int(image_size) // int(patch_size)
        w = int(image_size) // int(patch_size)
        self.pos = nn.Parameter(torch.zeros(1, h * w, int(dim)))
        dp_rates = torch.linspace(0.0, float(drop_path), steps=int(depth)).tolist()
        self.blocks = nn.Sequential(*[FastFormerBlock(int(dim), drop_path=float(dp_rates[i])) for i in range(int(depth))])
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
    "fastformer_vit_tiny": {"dim": 192, "depth": 10, "patch": 4},
    "fastformer_vit_small": {"dim": 256, "depth": 12, "patch": 4},
}


def build_fastformer_vit_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "fastformer_vit_tiny",
    image_size: int = 64,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown FastFormer-ViT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return FastFormerViTClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        patch_size=int(spec["patch"]),
        dim=int(spec["dim"]),
        depth=int(spec["depth"]),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 3, 64, 64)
    m = build_fastformer_vit_classifier(in_channels=3, num_classes=10, variant="fastformer_vit_tiny", image_size=64)
    y = m(x)
    print("fastformer_vit_tiny", tuple(y.shape))

