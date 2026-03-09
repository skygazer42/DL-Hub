
from dataclasses import dataclass

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(embed_dim // num_heads)

        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        self.out = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.dropout = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        scale = float(self.head_dim) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.out(out)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(int(embed_dim))
        self.attn = MultiHeadSelfAttention(embed_dim=int(embed_dim), num_heads=int(num_heads), dropout=dropout)
        self.drop1 = nn.Dropout(p=float(dropout))

        self.ln2 = nn.LayerNorm(int(embed_dim))
        self.ff = nn.Sequential(
            nn.Linear(int(embed_dim), int(ff_dim)),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(ff_dim), int(embed_dim)),
        )
        self.drop2 = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.ln1(x)))
        x = x + self.drop2(self.ff(self.ln2(x)))
        return x


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 64
    patch_size: int = 8
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 256
    dropout: float = 0.1
    num_classes: int = 4


class ViTClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if int(cfg.image_size) % int(cfg.patch_size) != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.cfg = cfg
        grid = int(cfg.image_size) // int(cfg.patch_size)
        num_patches = grid * grid

        self.patch_embed = nn.Conv2d(1, int(cfg.embed_dim), kernel_size=int(cfg.patch_size), stride=int(cfg.patch_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(cfg.embed_dim)))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, int(cfg.embed_dim)))
        self.drop = nn.Dropout(p=float(cfg.dropout))

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=int(cfg.embed_dim),
                    num_heads=int(cfg.num_heads),
                    ff_dim=int(cfg.ff_dim),
                    dropout=float(cfg.dropout),
                )
                for _ in range(int(cfg.num_layers))
            ]
        )
        self.ln = nn.LayerNorm(int(cfg.embed_dim))
        self.head = nn.Linear(int(cfg.embed_dim), int(cfg.num_classes))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        b = int(x.shape[0])

        tokens = self.patch_embed(x)  # (B, C, Gh, Gw)
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, N, C)

        cls = self.cls_token.expand(b, -1, -1)  # (B, 1, C)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+N, C)
        tokens = tokens + self.pos_embed
        tokens = self.drop(tokens)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.ln(tokens)
        cls_out = tokens[:, 0]
        return self.head(cls_out)


__all__ = ["ViTClassifier", "ModelConfig"]

