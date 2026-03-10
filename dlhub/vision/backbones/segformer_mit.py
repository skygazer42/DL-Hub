import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath, scale_channels


class SpatialReductionAttention(nn.Module):
    """SegFormer MiT-like attention with spatial reduction for K/V (simplified)."""

    def __init__(self, dim: int, num_heads: int, *, sr_ratio: int = 1) -> None:
        super().__init__()
        d = int(dim)
        h = int(num_heads)
        if d % h != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = d
        self.num_heads = h
        self.head_dim = d // h
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)
        self.sr_ratio = int(sr_ratio)
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(d, d, kernel_size=self.sr_ratio, stride=self.sr_ratio, bias=False)
            self.norm = nn.LayerNorm(d)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x: torch.Tensor, *, hw: tuple[int, int]) -> torch.Tensor:
        b, n, d = x.shape
        h, w = int(hw[0]), int(hw[1])
        if n != h * w:
            raise ValueError("hw mismatch")
        q = self.q(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        if self.sr is not None:
            x2d = x.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
            x2d = self.sr(x2d)
            x_sr = x2d.flatten(2).transpose(1, 2)
            x_sr = self.norm(x_sr)
            kv = self.kv(x_sr)
            nk = x_sr.shape[1]
        else:
            kv = self.kv(x)
            nk = n
        kv = kv.view(b, nk, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * float(self.scale)
        attn = torch.softmax(attn, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(b, n, d)
        return self.proj(y)


class MiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, *, sr_ratio: int, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = nn.LayerNorm(d)
        self.attn = SpatialReductionAttention(d, int(num_heads), sr_ratio=int(sr_ratio))
        self.dp1 = DropPath(float(drop_path))
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.dp2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor, *, hw: tuple[int, int]) -> torch.Tensor:
        x = x + self.dp1(self.attn(self.norm1(x), hw=hw))
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, *, kernel_size: int, stride: int) -> None:
        super().__init__()
        k = int(kernel_size)
        s = int(stride)
        p = k // 2
        self.proj = nn.Conv2d(
            int(in_ch), int(embed_dim), kernel_size=k, stride=s, padding=p, bias=True
        )
        self.norm = nn.LayerNorm(int(embed_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        x = self.proj(x)
        h, w = x.shape[-2], x.shape[-1]
        t = x.flatten(2).transpose(1, 2)
        t = self.norm(t)
        return t, (h, w)


class SegFormerMiTClassifier(nn.Module):
    """SegFormer MiT pyramid transformer (classifier head for zoo usage)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int, int] = (64, 128, 320, 512),
        depths: tuple[int, int, int, int] = (2, 2, 6, 2),
        heads: tuple[int, int, int, int] = (1, 2, 5, 8),
        sr: tuple[int, int, int, int] = (8, 4, 2, 1),
        width_mult: float = 1.0,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = tuple(scale_channels(int(d), float(width_mult), min_ch=16, divisor=8) for d in dims)
        depths = tuple(int(x) for x in depths)
        heads = tuple(int(h) for h in heads)
        sr = tuple(int(s) for s in sr)
        total = sum(depths)
        dp_rates = torch.linspace(0.0, float(drop_path), steps=total).tolist()
        dp_iter = iter(dp_rates)

        self.patch1 = OverlapPatchEmbed(int(in_channels), dims[0], kernel_size=7, stride=4)
        self.patch2 = OverlapPatchEmbed(dims[0], dims[1], kernel_size=3, stride=2)
        self.patch3 = OverlapPatchEmbed(dims[1], dims[2], kernel_size=3, stride=2)
        self.patch4 = OverlapPatchEmbed(dims[2], dims[3], kernel_size=3, stride=2)

        self.stage1 = nn.ModuleList(
            [
                MiTBlock(dims[0], heads[0], sr_ratio=sr[0], drop_path=float(next(dp_iter)))
                for _ in range(depths[0])
            ]
        )
        self.stage2 = nn.ModuleList(
            [
                MiTBlock(dims[1], heads[1], sr_ratio=sr[1], drop_path=float(next(dp_iter)))
                for _ in range(depths[1])
            ]
        )
        self.stage3 = nn.ModuleList(
            [
                MiTBlock(dims[2], heads[2], sr_ratio=sr[2], drop_path=float(next(dp_iter)))
                for _ in range(depths[2])
            ]
        )
        self.stage4 = nn.ModuleList(
            [
                MiTBlock(dims[3], heads[3], sr_ratio=sr[3], drop_path=float(next(dp_iter)))
                for _ in range(depths[3])
            ]
        )

        self.norm = nn.LayerNorm(dims[-1])
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(dims[-1], int(num_classes))

    def _run_stage(
        self, x: torch.Tensor, hw: tuple[int, int], blocks: nn.ModuleList
    ) -> torch.Tensor:
        for b in blocks:
            x = b(x, hw=hw)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        t, hw = self.patch1(x)
        t = self._run_stage(t, hw, self.stage1)
        x2d = t.transpose(1, 2).contiguous().view(t.shape[0], -1, hw[0], hw[1])

        t, hw = self.patch2(x2d)
        t = self._run_stage(t, hw, self.stage2)
        x2d = t.transpose(1, 2).contiguous().view(t.shape[0], -1, hw[0], hw[1])

        t, hw = self.patch3(x2d)
        t = self._run_stage(t, hw, self.stage3)
        x2d = t.transpose(1, 2).contiguous().view(t.shape[0], -1, hw[0], hw[1])

        t, hw = self.patch4(x2d)
        t = self._run_stage(t, hw, self.stage4)
        t = self.norm(t)
        t = self.drop(t.mean(dim=1))
        return self.head(t)


_VARIANTS: dict[str, dict] = {
    "mit_b0": {
        "dims": (32, 64, 160, 256),
        "depths": (2, 2, 2, 2),
        "heads": (1, 2, 5, 8),
        "sr": (8, 4, 2, 1),
    },
    "mit_b1": {
        "dims": (64, 128, 320, 512),
        "depths": (2, 2, 4, 2),
        "heads": (1, 2, 5, 8),
        "sr": (8, 4, 2, 1),
    },
    "mit_b2": {
        "dims": (64, 128, 320, 512),
        "depths": (3, 4, 6, 3),
        "heads": (1, 2, 5, 8),
        "sr": (8, 4, 2, 1),
    },
}


def build_segformer_mit_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "mit_b0",
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MiT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return SegFormerMiTClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dims=tuple(map(int, spec["dims"])),
        depths=tuple(map(int, spec["depths"])),
        heads=tuple(map(int, spec["heads"])),
        sr=tuple(map(int, spec["sr"])),
        width_mult=float(width_mult),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_segformer_mit_classifier(
        in_channels=3, num_classes=10, variant="mit_b0", width_mult=0.5
    )
    y = m(x)
    print("mit_b0", tuple(y.shape))
