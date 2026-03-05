from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ._utils import pad_to_multiple, unpad


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        h = int(hidden_dim)
        self.fc1 = nn.Linear(d, h)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(p=float(dropout))
        self.fc2 = nn.Linear(h, d)
        self.drop2 = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


class WindowAttention(nn.Module):
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
        # x: (B*nW, N, D)
        b, n, d = x.shape
        if d != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {d}")
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, Dh)
        attn = torch.matmul(q, k.transpose(-2, -1)) * float(self.scale)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, d)
        y = self.proj(y)
        return self.proj_drop(y)


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    # x: (B, H, W, C) -> (B*nW, ws*ws, C)
    b, h, w, c = x.shape
    ws = int(window_size)
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b * (h // ws) * (w // ws), ws * ws, c)


def _window_reverse(windows: torch.Tensor, window_size: int, *, b: int, h: int, w: int, c: int) -> torch.Tensor:
    # windows: (B*nW, ws*ws, C) -> (B, H, W, C)
    ws = int(window_size)
    x = windows.view(b, h // ws, w // ws, ws, ws, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, h, w, c)


class SwinIRBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        window_size: int = 8,
        shift: bool = False,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d = int(dim)
        self.window_size = int(window_size)
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        self.shift_size = self.window_size // 2 if bool(shift) else 0

        self.norm1 = nn.LayerNorm(d)
        self.attn = WindowAttention(d, int(num_heads), dropout=float(dropout))
        self.norm2 = nn.LayerNorm(d)
        self.mlp = MLP(d, int(round(d * float(mlp_ratio))), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        b, h, w, c = x.shape
        ws = int(self.window_size)
        if h % ws != 0 or w % ws != 0:
            raise ValueError("Input must be padded so H and W are multiples of window_size")

        shortcut = x
        if self.shift_size:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_w = _window_partition(x, ws)  # (B*nW, N, C)
        x_w = self.attn(self.norm1(x_w))
        x = _window_reverse(x_w, ws, b=b, h=h, w=w, c=c)

        if self.shift_size:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class SwinIR(nn.Module):
    """SwinIR-style denoiser (toy-first, pure torch).

    This is a compact window-attention model suitable for small images.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        embed_dim: int = 48,
        depth: int = 4,
        num_heads: int = 4,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        d = int(embed_dim)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if d <= 0:
            raise ValueError("embed_dim must be > 0")
        if d % int(num_heads) != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.window_size = int(window_size)
        self.in_conv = nn.Conv2d(c_in, d, kernel_size=3, padding=1, bias=True)
        blocks: list[nn.Module] = []
        for i in range(int(depth)):
            blocks.append(
                SwinIRBlock(
                    d,
                    int(num_heads),
                    window_size=int(window_size),
                    shift=bool(i % 2 == 1),
                    mlp_ratio=float(mlp_ratio),
                    dropout=0.0,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.mid_conv = nn.Conv2d(d, d, kernel_size=3, padding=1, bias=True)
        self.out_conv = nn.Conv2d(d, c_in, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        x_pad, pad_hw = pad_to_multiple(x, int(self.window_size), mode="reflect")
        inp = x_pad

        feat = self.in_conv(x_pad)
        b, c, h, w = feat.shape
        y = feat.permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)
        for blk in self.blocks:
            y = blk(y)
        y = y.permute(0, 3, 1, 2).contiguous()
        y = self.mid_conv(y) + feat
        out = inp + self.out_conv(y)
        return unpad(out, pad_hw)


_VARIANTS: dict[str, dict] = {
    "swinir_tiny": {"embed": 32, "depth": 2, "heads": 4, "window": 8, "mlp": 2.0},
    "swinir_small": {"embed": 48, "depth": 4, "heads": 4, "window": 8, "mlp": 2.0},
    "swinir_base": {"embed": 64, "depth": 6, "heads": 8, "window": 8, "mlp": 2.0},
}


def build_swinir_denoiser(
    *,
    in_channels: int,
    variant: str = "swinir_small",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SwinIR variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return SwinIR(
        in_channels=int(in_channels),
        embed_dim=int(spec["embed"]),
        depth=int(spec["depth"]),
        num_heads=int(spec["heads"]),
        window_size=int(spec["window"]),
        mlp_ratio=float(spec["mlp"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_swinir_denoiser(in_channels=1, variant="swinir_tiny")
    y = m(noisy)
    print("swinir_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")

