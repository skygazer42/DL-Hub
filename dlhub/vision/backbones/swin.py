from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """x: (B, H, W, C) -> windows: (B*nW, Ws*Ws, C)."""

    b, h, w, c = x.shape
    ws = int(window_size)
    if ws <= 0:
        raise ValueError("window_size must be > 0")
    if h % ws != 0 or w % ws != 0:
        raise ValueError("H and W must be divisible by window_size")

    x = x.view(b, h // ws, ws, w // ws, ws, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, H/ws, W/ws, ws, ws, C)
    windows = x.view(b * (h // ws) * (w // ws), ws * ws, c)
    return windows


def _window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """windows: (B*nW, Ws*Ws, C) -> x: (B, H, W, C)."""

    ws = int(window_size)
    if ws <= 0:
        raise ValueError("window_size must be > 0")
    n_windows = (int(h) // ws) * (int(w) // ws)
    b = int(windows.shape[0] // n_windows)
    c = int(windows.shape[-1])

    x = windows.view(b, int(h) // ws, int(w) // ws, ws, ws, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(b, int(h), int(w), c)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if int(dim) % int(num_heads) != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(dim // num_heads)
        self.scale = float(self.head_dim) ** -0.5

        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=False)
        self.proj = nn.Linear(self.dim, self.dim, bias=False)
        self.dropout = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        """x: (B*nW, N, C)"""

        b, n, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        if attn_mask is not None:
            # attn_mask: (nW, N, N). Broadcast it over the batch dimension.
            nW = int(attn_mask.shape[0])
            if b % nW != 0:
                raise ValueError("attn_mask nW must divide the batch of windows")
            scores = scores.view(b // nW, nW, self.num_heads, n, n)
            scores = scores + attn_mask.to(device=scores.device, dtype=scores.dtype).view(1, nW, 1, n, n)
            scores = scores.view(b, self.num_heads, n, n)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, N, D)
        out = out.transpose(1, 2).contiguous().view(b, n, c)
        out = self.proj(out)
        return out


class SwinBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.input_resolution = (int(input_resolution[0]), int(input_resolution[1]))
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.mlp_ratio = float(mlp_ratio)

        h, w = self.input_resolution
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        if self.shift_size < 0 or self.shift_size >= self.window_size:
            raise ValueError("shift_size must be in [0, window_size)")
        if h % self.window_size != 0 or w % self.window_size != 0:
            raise ValueError("input resolution must be divisible by window_size")

        self.ln1 = nn.LayerNorm(self.dim)
        self.attn = WindowAttention(dim=self.dim, num_heads=self.num_heads, dropout=dropout)
        self.drop1 = nn.Dropout(p=float(dropout))

        self.ln2 = nn.LayerNorm(self.dim)
        hidden = int(self.dim * self.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, hidden),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden, self.dim),
        )
        self.drop2 = nn.Dropout(p=float(dropout))

        self.register_buffer("attn_mask", self._build_attn_mask(), persistent=False)

    def _build_attn_mask(self) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None

        h, w = self.input_resolution
        ws = self.window_size
        ss = self.shift_size

        img_mask = torch.zeros((1, h, w, 1), dtype=torch.int64)
        cnt = 0
        for y in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
            for x in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
                img_mask[:, y, x, :] = cnt
                cnt += 1

        mask_windows = _window_partition(img_mask, window_size=ws)  # (nW, N, 1)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, N, N)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C)"""

        b, h, w, c = x.shape
        if (h, w) != self.input_resolution:
            raise ValueError(f"Expected resolution {self.input_resolution}, got {(h, w)}")

        shortcut = x
        x = self.ln1(x).view(b, h, w, c)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = _window_partition(x, window_size=self.window_size)  # (B*nW, N, C)
        windows_out = self.attn(windows, attn_mask=self.attn_mask)
        windows_out = self.drop1(windows_out)

        x = _window_reverse(windows_out, window_size=self.window_size, h=h, w=w)  # (B, H, W, C)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + x
        x = x + self.drop2(self.mlp(self.ln2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution: tuple[int, int], dim: int) -> None:
        super().__init__()
        h, w = int(input_resolution[0]), int(input_resolution[1])
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError("PatchMerging requires even H and W")
        self.input_resolution = (h, w)
        self.dim = int(dim)
        self.reduction = nn.Linear(4 * self.dim, 2 * self.dim, bias=False)
        self.norm = nn.LayerNorm(4 * self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, C) -> (B, H/2, W/2, 2C)"""

        b, h, w, c = x.shape
        if (h, w) != self.input_resolution:
            raise ValueError(f"Expected resolution {self.input_resolution}, got {(h, w)}")
        if c != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got c={c}")

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinStage(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
        downsample: bool,
    ) -> None:
        super().__init__()
        h, w = int(input_resolution[0]), int(input_resolution[1])
        ws = int(window_size)
        if ws <= 0:
            raise ValueError("window_size must be > 0")
        if h % ws != 0 or w % ws != 0:
            raise ValueError("input resolution must be divisible by window_size")

        blocks: list[nn.Module] = []
        for i in range(int(depth)):
            shift_size = 0 if (i % 2 == 0) else ws // 2
            blocks.append(
                SwinBlock(
                    dim=int(dim),
                    input_resolution=(h, w),
                    num_heads=int(num_heads),
                    window_size=ws,
                    shift_size=int(shift_size),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.downsample = PatchMerging((h, w), dim=int(dim)) if bool(downsample) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


@dataclass(frozen=True)
class SwinConfig:
    image_size: int
    patch_size: int
    in_channels: int
    embed_dim: int
    depths: tuple[int, int, int, int]
    num_heads: tuple[int, int, int, int]
    window_sizes: tuple[int, int, int, int]
    mlp_ratio: float
    dropout: float
    num_classes: int


class SwinClassifier(nn.Module):
    def __init__(self, cfg: SwinConfig) -> None:
        super().__init__()
        if int(cfg.image_size) % int(cfg.patch_size) != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.cfg = cfg
        grid = int(cfg.image_size) // int(cfg.patch_size)

        self.patch_embed = nn.Conv2d(
            int(cfg.in_channels),
            int(cfg.embed_dim),
            kernel_size=int(cfg.patch_size),
            stride=int(cfg.patch_size),
        )

        self.pos_drop = nn.Dropout(p=float(cfg.dropout))

        dims = [int(cfg.embed_dim), int(cfg.embed_dim) * 2, int(cfg.embed_dim) * 4, int(cfg.embed_dim) * 8]
        resolutions = [(grid, grid), (grid // 2, grid // 2), (grid // 4, grid // 4), (grid // 8, grid // 8)]

        stages: list[nn.Module] = []
        for i in range(4):
            stages.append(
                SwinStage(
                    dim=dims[i],
                    input_resolution=resolutions[i],
                    depth=int(cfg.depths[i]),
                    num_heads=int(cfg.num_heads[i]),
                    window_size=int(cfg.window_sizes[i]),
                    mlp_ratio=float(cfg.mlp_ratio),
                    dropout=float(cfg.dropout),
                    downsample=(i < 3),
                )
            )
        self.stages = nn.Sequential(*stages)

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], int(cfg.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch_embed(x)  # (B, C, Gh, Gw)
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, Gh, Gw, C)
        x = self.pos_drop(x)
        x = self.stages(x)  # (B, h, w, c)
        x = x.mean(dim=(1, 2))
        x = self.norm(x)
        return self.head(x)


def build_swin_classifier(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    w = float(width_mult)

    if name in {"swin_tiny", "tiny"}:
        embed_dim = max(32, int(round(64 * w)))
        depths = (2, 2, 4, 2)
        num_heads = (2, 4, 8, 16)
    elif name in {"swin_small", "small"}:
        embed_dim = max(32, int(round(72 * w)))
        depths = (2, 2, 6, 2)
        num_heads = (3, 6, 12, 24)
    elif name in {"swin_base", "base"}:
        embed_dim = max(32, int(round(96 * w)))
        depths = (2, 2, 6, 2)
        num_heads = (3, 6, 12, 24)
    elif name in {"swin_tiny_w2"}:
        embed_dim = max(32, int(round(64 * w)))
        depths = (2, 2, 4, 2)
        num_heads = (2, 4, 8, 16)
    else:
        raise ValueError("Unknown Swin variant. Supported: swin_tiny|swin_small|swin_base")

    # For small images (e.g. 64x64), the final stage can become very small.
    # Use per-stage window sizes to ensure divisibility.
    patch_size = 4
    grid = int(image_size) // int(patch_size)
    ws1 = 4 if grid % 4 == 0 else 2
    ws2 = 4 if (grid // 2) % 4 == 0 else 2
    ws3 = 2 if (grid // 4) % 2 == 0 else 1
    ws4 = 1 if (grid // 8) <= 1 else 2
    window_sizes = (ws1, ws2, ws3, ws4)

    return SwinClassifier(
        SwinConfig(
            image_size=int(image_size),
            patch_size=int(patch_size),
            in_channels=int(in_channels),
            embed_dim=int(embed_dim),
            depths=tuple(map(int, depths)),
            num_heads=tuple(map(int, num_heads)),
            window_sizes=tuple(map(int, window_sizes)),
            mlp_ratio=4.0,
            dropout=float(dropout),
            num_classes=int(num_classes),
        )
    )


__all__ = ["build_swin_classifier"]

