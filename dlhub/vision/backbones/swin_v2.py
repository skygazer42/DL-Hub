from __future__ import annotations

import math

import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath, GlobalAvgPoolHead, LayerNorm2d, scale_channels
from dlhub.vision.backbones._transformer import MLP, MultiheadSelfAttention


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    # x: (B, H, W, C) -> (B*num_windows, ws*ws, C)
    b, h, w, c = x.shape
    ws = int(window_size)
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b * (h // ws) * (w // ws), ws * ws, c)
    return x


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    # windows: (B*num_windows, ws*ws, C) -> (B, H, W, C)
    ws = int(window_size)
    b = int(windows.shape[0] // (h // ws * w // ws))
    x = windows.view(b, h // ws, w // ws, ws, ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class SwinV2Block(nn.Module):
    """Window attention block (no shift, simplified)."""

    def __init__(self, dim: int, num_heads: int, *, window_size: int = 8, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        self.window_size = int(window_size)
        self.norm1 = nn.LayerNorm(d)
        self.attn = MultiheadSelfAttention(d, int(num_heads), dropout=0.0)
        self.dp1 = DropPath(float(drop_path))
        self.norm2 = nn.LayerNorm(d)
        self.mlp = MLP(d, 4 * d, dropout=0.0, act="gelu")
        self.dp2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor, *, hw: tuple[int, int]) -> torch.Tensor:
        b, n, d = x.shape
        h, w = int(hw[0]), int(hw[1])
        ws = self.window_size
        if n != h * w:
            raise ValueError("hw mismatch")
        if h % ws != 0 or w % ws != 0:
            # pad (rare for our smokes)
            x2d = x.view(b, h, w, d)
            pad_h = (ws - h % ws) % ws
            pad_w = (ws - w % ws) % ws
            x2d = nn.functional.pad(x2d, (0, 0, 0, pad_w, 0, pad_h))
            h2, w2 = x2d.shape[1], x2d.shape[2]
        else:
            x2d = x.view(b, h, w, d)
            h2, w2 = h, w

        windows = window_partition(x2d, ws)
        windows = windows + self.dp1(self.attn(self.norm1(windows)))
        windows = windows + self.dp2(self.mlp(self.norm2(windows)))
        x2d = window_reverse(windows, ws, h2, w2)
        x2d = x2d[:, :h, :w]
        return x2d.reshape(b, h * w, d)


class PatchMerging(nn.Module):
    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.out_dim = int(out_dim)
        self.norm = nn.LayerNorm(4 * self.dim)
        self.proj = nn.Linear(4 * self.dim, self.out_dim)

    def forward(self, x: torch.Tensor, *, hw: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
        b, n, d = x.shape
        h, w = int(hw[0]), int(hw[1])
        x = x.view(b, h, w, d)
        if h % 2 != 0 or w % 2 != 0:
            x = x[:, : h - (h % 2), : w - (w % 2)]
            h, w = x.shape[1], x.shape[2]
        x0 = x[:, 0::2, 0::2]
        x1 = x[:, 0::2, 1::2]
        x2 = x[:, 1::2, 0::2]
        x3 = x[:, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=-1).view(b, (h // 2) * (w // 2), 4 * d)
        x = self.norm(x)
        x = self.proj(x)
        return x, (h // 2, w // 2)


class SwinV2Classifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int = 64,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: tuple[int, int, int, int] = (2, 2, 6, 2),
        heads: tuple[int, int, int, int] = (3, 6, 12, 24),
        window_size: int = 8,
        width_mult: float = 1.0,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.window_size = int(window_size)
        d0 = scale_channels(int(embed_dim), float(width_mult), min_ch=16, divisor=8)
        dims = (d0, 2 * d0, 4 * d0, 8 * d0)
        depths = tuple(int(x) for x in depths)
        heads = tuple(int(h) for h in heads)
        total = sum(depths)
        dp_rates = torch.linspace(0.0, float(drop_path), steps=total).tolist()
        dp_iter = iter(dp_rates)

        self.patch = nn.Conv2d(int(in_channels), dims[0], kernel_size=self.patch_size, stride=self.patch_size, bias=True)
        self.hw = (self.image_size // self.patch_size, self.image_size // self.patch_size)

        self.stage1 = nn.ModuleList([SwinV2Block(dims[0], heads[0], window_size=self.window_size, drop_path=float(next(dp_iter))) for _ in range(depths[0])])
        self.merge1 = PatchMerging(dims[0], dims[1])
        self.stage2 = nn.ModuleList([SwinV2Block(dims[1], heads[1], window_size=self.window_size, drop_path=float(next(dp_iter))) for _ in range(depths[1])])
        self.merge2 = PatchMerging(dims[1], dims[2])
        self.stage3 = nn.ModuleList([SwinV2Block(dims[2], heads[2], window_size=self.window_size, drop_path=float(next(dp_iter))) for _ in range(depths[2])])
        self.merge3 = PatchMerging(dims[2], dims[3])
        self.stage4 = nn.ModuleList([SwinV2Block(dims[3], heads[3], window_size=self.window_size, drop_path=float(next(dp_iter))) for _ in range(depths[3])])

        self.norm = nn.LayerNorm(dims[-1])
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(dims[-1], int(num_classes))

    def _run(self, blocks: nn.ModuleList, x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        for b in blocks:
            x = b(x, hw=hw)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch(x).flatten(2).transpose(1, 2)
        hw = self.hw
        x = self._run(self.stage1, x, hw)
        x, hw = self.merge1(x, hw=hw)
        x = self._run(self.stage2, x, hw)
        x, hw = self.merge2(x, hw=hw)
        x = self._run(self.stage3, x, hw)
        x, hw = self.merge3(x, hw=hw)
        x = self._run(self.stage4, x, hw)
        x = self.norm(x)
        x = self.drop(x.mean(dim=1))
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "swin_v2_tiny": {"embed": 96, "depths": (2, 2, 6, 2), "heads": (3, 6, 12, 24), "ws": 8},
    "swin_v2_small": {"embed": 96, "depths": (2, 2, 18, 2), "heads": (3, 6, 12, 24), "ws": 8},
}


def build_swin_v2_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "swin_v2_tiny",
    image_size: int = 64,
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SwinV2 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return SwinV2Classifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        patch_size=4,
        embed_dim=int(spec["embed"]),
        depths=tuple(map(int, spec["depths"])),
        heads=tuple(map(int, spec["heads"])),
        window_size=int(spec["ws"]),
        width_mult=float(width_mult),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_swin_v2_classifier(in_channels=3, num_classes=10, variant="swin_v2_tiny", image_size=64, width_mult=0.5)
    y = m(x)
    print("swin_v2_tiny", tuple(y.shape))

