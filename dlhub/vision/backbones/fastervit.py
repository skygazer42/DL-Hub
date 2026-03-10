import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath, scale_channels
from dlhub.vision.backbones._transformer import MLP, MultiheadSelfAttention


def _window_partition(x: torch.Tensor, window: int) -> torch.Tensor:
    b, h, w, c = x.shape
    ws = int(window)
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b * (h // ws) * (w // ws), ws * ws, c)
    return x


def _window_reverse(windows: torch.Tensor, window: int, h: int, w: int) -> torch.Tensor:
    ws = int(window)
    b = int(windows.shape[0] // (h // ws * w // ws))
    x = windows.view(b, h // ws, w // ws, ws, ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class FasterViTBlock(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, *, window: int = 8, drop_path: float = 0.0
    ) -> None:
        super().__init__()
        d = int(dim)
        self.window = int(window)
        self.norm1 = nn.LayerNorm(d)
        self.attn = MultiheadSelfAttention(d, int(num_heads), dropout=0.0)
        self.dp1 = DropPath(float(drop_path))
        self.norm2 = nn.LayerNorm(d)
        self.mlp = MLP(d, 4 * d, dropout=0.0, act="gelu")
        self.dp2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor, *, hw: tuple[int, int]) -> torch.Tensor:
        b, n, d = x.shape
        h, w = int(hw[0]), int(hw[1])
        ws = self.window
        if n != h * w:
            raise ValueError("hw mismatch")
        x2d = x.view(b, h, w, d)
        windows = _window_partition(x2d, ws)
        windows = windows + self.dp1(self.attn(self.norm1(windows)))
        windows = windows + self.dp2(self.mlp(self.norm2(windows)))
        x2d = _window_reverse(windows, ws, h, w)
        return x2d.view(b, n, d)


class PatchMerging(nn.Module):
    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.out_dim = int(out_dim)
        self.norm = nn.LayerNorm(4 * self.dim)
        self.proj = nn.Linear(4 * self.dim, self.out_dim)

    def forward(
        self, x: torch.Tensor, *, hw: tuple[int, int]
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        b, n, d = x.shape
        h, w = int(hw[0]), int(hw[1])
        x = x.view(b, h, w, d)
        x0 = x[:, 0::2, 0::2]
        x1 = x[:, 0::2, 1::2]
        x2 = x[:, 1::2, 0::2]
        x3 = x[:, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=-1).view(b, (h // 2) * (w // 2), 4 * d)
        x = self.norm(x)
        x = self.proj(x)
        return x, (h // 2, w // 2)


class FasterViTClassifier(nn.Module):
    """FasterViT (simplified).

    Implemented as a TinyViT-like hierarchical window transformer with different
    presets. Each algorithm-family file is self-contained.
    """

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
        window: int = 8,
        width_mult: float = 1.0,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d0 = scale_channels(int(embed_dim), float(width_mult), min_ch=16, divisor=8)
        dims = (d0, 2 * d0, 4 * d0, 8 * d0)
        depths = tuple(int(x) for x in depths)
        heads = tuple(int(h) for h in heads)
        total = sum(depths)
        dp_rates = torch.linspace(0.0, float(drop_path), steps=total).tolist()
        dp_iter = iter(dp_rates)

        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.hw = (self.image_size // self.patch_size, self.image_size // self.patch_size)
        self.patch = nn.Conv2d(
            int(in_channels),
            dims[0],
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        self.stage1 = nn.ModuleList(
            [
                FasterViTBlock(
                    dims[0], heads[0], window=int(window), drop_path=float(next(dp_iter))
                )
                for _ in range(depths[0])
            ]
        )
        self.merge1 = PatchMerging(dims[0], dims[1])
        self.stage2 = nn.ModuleList(
            [
                FasterViTBlock(
                    dims[1], heads[1], window=int(window), drop_path=float(next(dp_iter))
                )
                for _ in range(depths[1])
            ]
        )
        self.merge2 = PatchMerging(dims[1], dims[2])
        self.stage3 = nn.ModuleList(
            [
                FasterViTBlock(
                    dims[2], heads[2], window=int(window), drop_path=float(next(dp_iter))
                )
                for _ in range(depths[2])
            ]
        )
        self.merge3 = PatchMerging(dims[2], dims[3])
        self.stage4 = nn.ModuleList(
            [
                FasterViTBlock(
                    dims[3], heads[3], window=int(window), drop_path=float(next(dp_iter))
                )
                for _ in range(depths[3])
            ]
        )

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
    "fastervit_t0": {"embed": 96, "depths": (2, 2, 6, 2), "heads": (3, 6, 12, 24), "window": 8},
    "fastervit_t1": {"embed": 128, "depths": (2, 2, 12, 2), "heads": (4, 8, 16, 32), "window": 8},
}


def build_fastervit_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "fastervit_t0",
    image_size: int = 64,
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown FasterViT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return FasterViTClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        patch_size=4,
        embed_dim=int(spec["embed"]),
        depths=tuple(map(int, spec["depths"])),
        heads=tuple(map(int, spec["heads"])),
        window=int(spec["window"]),
        width_mult=float(width_mult),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_fastervit_classifier(
        in_channels=3, num_classes=10, variant="fastervit_t0", image_size=64, width_mult=0.5
    )
    y = m(x)
    print("fastervit_t0", tuple(y.shape))
