from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath
from dlhub.vision.backbones._transformer import MLP, PatchEmbed


class LinformerSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, *, num_tokens: int, proj_tokens: int) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        n = int(num_tokens)
        k = int(proj_tokens)
        if d % h != 0:
            raise ValueError("dim must be divisible by num_heads")
        if k <= 0 or k > n:
            raise ValueError("proj_tokens must be in (0, num_tokens]")
        self.dim = d
        self.num_heads = h
        self.head_dim = d // h
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        # projection along token dimension for each head
        self.E = nn.Parameter(torch.randn(h, k, n) * 0.02)
        self.F = nn.Parameter(torch.randn(h, k, n) * 0.02)
        self.proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q = self.q(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,N,Dh)
        kv = self.kv(x).view(b, n, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B,H,N,Dh)
        # project tokens: (H,K,N) @ (B,H,N,Dh) -> (B,H,K,Dh)
        k = torch.einsum("hkn,bhnd->bhkd", self.E, k)
        v = torch.einsum("hkn,bhnd->bhkd", self.F, v)
        attn = (q @ k.transpose(-2, -1)) * float(self.scale)  # (B,H,N,K)
        attn = torch.softmax(attn, dim=-1)
        y = attn @ v  # (B,H,N,Dh)
        y = y.transpose(1, 2).contiguous().view(b, n, d)
        return self.proj(y)


class LinformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, *, num_tokens: int, proj_tokens: int, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = nn.LayerNorm(d)
        self.attn = LinformerSelfAttention(d, int(num_heads), num_tokens=int(num_tokens), proj_tokens=int(proj_tokens))
        self.dp1 = DropPath(float(drop_path))
        self.norm2 = nn.LayerNorm(d)
        self.mlp = MLP(d, 4 * d, dropout=0.0, act="gelu")
        self.dp2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.attn(self.norm1(x)))
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


class LinformerV2Classifier(nn.Module):
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
        proj_tokens: int = 64,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch = PatchEmbed(int(in_channels), int(dim), patch_size=int(patch_size))
        h = int(image_size) // int(patch_size)
        w = int(image_size) // int(patch_size)
        n = h * w
        self.pos = nn.Parameter(torch.zeros(1, n, int(dim)))
        dp_rates = torch.linspace(0.0, float(drop_path), steps=int(depth)).tolist()
        self.blocks = nn.Sequential(
            *[LinformerBlock(int(dim), int(heads), num_tokens=n, proj_tokens=int(proj_tokens), drop_path=float(dp_rates[i])) for i in range(int(depth))]
        )
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
    "linformer_v2_tiny": {"dim": 192, "depth": 8, "heads": 6, "patch": 4, "proj": 64},
    "linformer_v2_small": {"dim": 256, "depth": 10, "heads": 8, "patch": 4, "proj": 64},
}


def build_linformer_v2_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "linformer_v2_tiny",
    image_size: int = 64,
    proj_tokens: int | None = None,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Linformer-v2 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return LinformerV2Classifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        patch_size=int(spec["patch"]),
        dim=int(spec["dim"]),
        depth=int(spec["depth"]),
        heads=int(spec["heads"]),
        proj_tokens=int(spec["proj"] if proj_tokens is None else proj_tokens),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 3, 64, 64)
    m = build_linformer_v2_classifier(in_channels=3, num_classes=10, variant="linformer_v2_tiny", image_size=64)
    y = m(x)
    print("linformer_v2_tiny", tuple(y.shape))

