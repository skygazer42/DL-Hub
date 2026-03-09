
import math

import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        dropout: float = 0.0,
        act: str = "gelu",
    ) -> None:
        super().__init__()
        d = int(dim)
        h = int(hidden_dim)

        act_name = str(act).lower().strip()
        if act_name == "gelu":
            act_layer: nn.Module = nn.GELU()
        elif act_name in {"silu", "swish"}:
            act_layer = nn.SiLU(inplace=True)
        elif act_name == "relu":
            act_layer = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {act!r}")

        self.fc1 = nn.Linear(d, h)
        self.act = act_layer
        self.drop1 = nn.Dropout(p=float(dropout))
        self.fc2 = nn.Linear(h, d)
        self.drop2 = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        if d <= 0:
            raise ValueError("dim must be > 0")
        if h <= 0:
            raise ValueError("num_heads must be > 0")
        if d % h != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = d
        self.num_heads = h
        self.head_dim = d // h
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(d, 3 * d, bias=True)
        self.attn_drop = nn.Dropout(p=float(dropout))
        self.proj = nn.Linear(d, d, bias=True)
        self.proj_drop = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        b, n, d = x.shape
        if d != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {d}")

        qkv = self.qkv(x)  # (B, N, 3D)
        qkv = qkv.view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, H, N, Dh)

        attn = torch.matmul(q, k.transpose(-2, -1)) * float(self.scale)  # (B, H, N, N)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        y = torch.matmul(attn, v)  # (B, H, N, Dh)
        y = y.transpose(1, 2).contiguous().view(b, n, d)  # (B, N, D)
        y = self.proj(y)
        return self.proj_drop(y)


class TransformerEncoderBlock(nn.Module):
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
        self.attn = MultiheadSelfAttention(d, int(num_heads), dropout=float(dropout))
        self.drop_path1 = DropPath(float(drop_path))

        self.norm2 = nn.LayerNorm(d)
        mlp_hidden = int(round(d * float(mlp_ratio)))
        self.mlp = MLP(d, mlp_hidden, dropout=float(dropout), act="gelu")
        self.drop_path2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, *, patch_size: int) -> None:
        super().__init__()
        p = int(patch_size)
        if p <= 0:
            raise ValueError("patch_size must be > 0")
        self.patch_size = p
        self.proj = nn.Conv2d(int(in_channels), int(embed_dim), kernel_size=p, stride=p, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


def sinusoidal_positional_embedding(num_tokens: int, dim: int, *, device: torch.device) -> torch.Tensor:
    n = int(num_tokens)
    d = int(dim)
    if n <= 0:
        raise ValueError("num_tokens must be > 0")
    if d <= 0:
        raise ValueError("dim must be > 0")
    if d % 2 != 0:
        raise ValueError("dim must be even for sinusoidal embedding")

    pe = torch.zeros(n, d, device=device)
    pos = torch.arange(0, n, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

