
from dataclasses import dataclass

import torch
from torch import nn


def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """x: (B, H, W, C) -> windows: (B*nW, Ws*Ws, C)."""

    b, h, w, c = x.shape
    ws = int(window_size)
    if h % ws != 0 or w % ws != 0:
        raise ValueError("H and W must be divisible by window_size")

    x = x.view(b, h // ws, ws, w // ws, ws, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, H/ws, W/ws, ws, ws, C)
    windows = x.view(b * (h // ws) * (w // ws), ws * ws, c)
    return windows


def _window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """windows: (B*nW, Ws*Ws, C) -> x: (B, H, W, C)."""

    ws = int(window_size)
    n_windows = (h // ws) * (w // ws)
    b = int(windows.shape[0] // n_windows)
    c = int(windows.shape[-1])

    x = windows.view(b, h // ws, w // ws, ws, ws, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(b, h, w, c)
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


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 64
    patch_size: int = 4
    embed_dim: int = 96
    num_heads: int = 4
    depth: int = 4
    window_size: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    num_classes: int = 4


class SwinTinyClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if int(cfg.image_size) % int(cfg.patch_size) != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.cfg = cfg
        h = int(cfg.image_size) // int(cfg.patch_size)
        w = int(cfg.image_size) // int(cfg.patch_size)

        self.patch_embed = nn.Conv2d(1, int(cfg.embed_dim), kernel_size=int(cfg.patch_size), stride=int(cfg.patch_size))
        self.pos_drop = nn.Dropout(p=float(cfg.dropout))

        blocks: list[SwinBlock] = []
        for i in range(int(cfg.depth)):
            shift = 0 if (i % 2 == 0) else int(cfg.window_size) // 2
            blocks.append(
                SwinBlock(
                    dim=int(cfg.embed_dim),
                    input_resolution=(h, w),
                    num_heads=int(cfg.num_heads),
                    window_size=int(cfg.window_size),
                    shift_size=int(shift),
                    mlp_ratio=float(cfg.mlp_ratio),
                    dropout=float(cfg.dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.ln = nn.LayerNorm(int(cfg.embed_dim))
        self.head = nn.Linear(int(cfg.embed_dim), int(cfg.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        feats = self.patch_embed(x)  # (B, C, H', W')
        feats = feats.permute(0, 2, 3, 1).contiguous()  # (B, H', W', C)
        feats = self.pos_drop(feats)

        for blk in self.blocks:
            feats = blk(feats)

        feats = self.ln(feats)  # (B, H', W', C)
        pooled = feats.mean(dim=(1, 2))  # (B, C)
        return self.head(pooled)


__all__ = ["SwinTinyClassifier", "ModelConfig"]
