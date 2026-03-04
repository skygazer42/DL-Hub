from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._transformer import PatchEmbed, TransformerEncoderBlock


class CrossAttention(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, num_heads: int) -> None:
        super().__init__()
        qd = int(q_dim)
        kd = int(kv_dim)
        h = int(num_heads)
        if qd % h != 0:
            raise ValueError("q_dim must be divisible by num_heads")
        self.q_dim = qd
        self.kv_dim = kd
        self.num_heads = h
        self.head_dim = qd // h
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(qd, qd)
        self.k = nn.Linear(kd, qd)
        self.v = nn.Linear(kd, qd)
        self.proj = nn.Linear(qd, qd)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        b, nq, _ = q.shape
        _, nk, _ = kv.shape
        q = self.q(q).view(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(kv).view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(kv).view(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * float(self.scale)
        attn = torch.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(b, nq, self.q_dim)
        return self.proj(y)


class PerceiverIOClassifier(nn.Module):
    """Perceiver IO (simplified): encoder to latents + decoder query for classification."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        patch_size: int = 4,
        input_dim: int = 192,
        latent_dim: int = 256,
        num_latents: int = 64,
        num_heads: int = 8,
        depth: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = PatchEmbed(int(in_channels), int(input_dim), patch_size=int(patch_size))
        self.latents = nn.Parameter(torch.randn(1, int(num_latents), int(latent_dim)) * 0.02)
        self.cross_in = CrossAttention(int(latent_dim), int(input_dim), int(num_heads))
        self.self_lat = nn.Sequential(*[TransformerEncoderBlock(int(latent_dim), int(num_heads), mlp_ratio=4.0, dropout=0.0, drop_path=0.0) for _ in range(int(depth))])
        self.query = nn.Parameter(torch.randn(1, 1, int(latent_dim)) * 0.02)
        self.cross_out = CrossAttention(int(latent_dim), int(latent_dim), int(num_heads))
        self.norm = nn.LayerNorm(int(latent_dim))
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(int(latent_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        inp = self.embed(x)
        lat = self.latents.expand(inp.shape[0], -1, -1)
        lat = lat + self.cross_in(lat, inp)
        lat = self.self_lat(lat)
        q = self.query.expand(inp.shape[0], -1, -1)
        q = q + self.cross_out(q, lat)
        q = self.norm(q)
        q = self.drop(q.squeeze(1))
        return self.head(q)


_VARIANTS: dict[str, dict] = {
    "perceiver_io_tiny": {"input_dim": 192, "latent_dim": 256, "latents": 64, "heads": 8, "depth": 4},
    "perceiver_io_base": {"input_dim": 192, "latent_dim": 384, "latents": 128, "heads": 12, "depth": 6},
}


def build_perceiver_io_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "perceiver_io_tiny",
    patch_size: int = 4,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown PerceiverIO variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return PerceiverIOClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        patch_size=int(patch_size),
        input_dim=int(spec["input_dim"]),
        latent_dim=int(spec["latent_dim"]),
        num_latents=int(spec["latents"]),
        num_heads=int(spec["heads"]),
        depth=int(spec["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_perceiver_io_classifier(in_channels=3, num_classes=10, variant="perceiver_io_tiny")
    y = m(x)
    print("perceiver_io_tiny", tuple(y.shape))
