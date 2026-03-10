import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels
from dlhub.vision.backbones._transformer import TransformerEncoderBlock


class PatchEmbed2D(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, *, patch_size: int) -> None:
        super().__init__()
        p = int(patch_size)
        self.proj = nn.Conv2d(int(in_ch), int(embed_dim), kernel_size=p, stride=p, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        x = self.proj(x)
        h, w = int(x.shape[-2]), int(x.shape[-1])
        t = x.flatten(2).transpose(1, 2).contiguous()
        return t, (h, w)


class ConvPosEnc(nn.Module):
    def __init__(self, dim: int, *, kernel_size: int = 3) -> None:
        super().__init__()
        d = int(dim)
        k = int(kernel_size)
        self.dw = nn.Conv2d(d, d, kernel_size=k, padding=k // 2, groups=d, bias=True)

    def forward(self, t: torch.Tensor, *, hw: tuple[int, int]) -> torch.Tensor:
        h, w = int(hw[0]), int(hw[1])
        b, n, c = t.shape
        if n != h * w:
            raise ValueError("hw mismatch")
        x = t.transpose(1, 2).contiguous().view(b, c, h, w)
        x = x + self.dw(x)
        return x.flatten(2).transpose(1, 2).contiguous()


class TransNeXtBlock(nn.Module):
    def __init__(self, dim: int, heads: int, *, drop_path: float, dropout: float) -> None:
        super().__init__()
        self.cpe = ConvPosEnc(int(dim), kernel_size=3)
        self.block = TransformerEncoderBlock(
            int(dim), int(heads), mlp_ratio=4.0, dropout=float(dropout), drop_path=float(drop_path)
        )

    def forward(self, t: torch.Tensor, *, hw: tuple[int, int]) -> torch.Tensor:
        t = self.cpe(t, hw=hw)
        return self.block(t)


class TransNeXtClassifier(nn.Module):
    """TransNeXt (simplified): hierarchical patch embedding + transformer blocks + CPE."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int, int],
        depths: tuple[int, int, int, int],
        heads: tuple[int, int, int, int],
        patch_size: int = 4,
        width_mult: float = 1.0,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = tuple(scale_channels(int(d), float(width_mult), min_ch=16, divisor=8) for d in dims)
        depths = tuple(int(d) for d in depths)
        heads = tuple(int(h) for h in heads)
        total = sum(depths)
        dp_rates = torch.linspace(0.0, float(drop_path), steps=max(1, total)).tolist()
        dp_iter = iter(dp_rates)

        self.embed1 = PatchEmbed2D(int(in_channels), dims[0], patch_size=int(patch_size))
        self.stage1 = nn.ModuleList(
            [
                TransNeXtBlock(
                    dims[0], heads[0], drop_path=float(next(dp_iter, 0.0)), dropout=float(dropout)
                )
                for _ in range(depths[0])
            ]
        )

        self.down2 = nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)
        self.stage2 = nn.ModuleList(
            [
                TransNeXtBlock(
                    dims[1], heads[1], drop_path=float(next(dp_iter, 0.0)), dropout=float(dropout)
                )
                for _ in range(depths[1])
            ]
        )

        self.down3 = nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2)
        self.stage3 = nn.ModuleList(
            [
                TransNeXtBlock(
                    dims[2], heads[2], drop_path=float(next(dp_iter, 0.0)), dropout=float(dropout)
                )
                for _ in range(depths[2])
            ]
        )

        self.down4 = nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2)
        self.stage4 = nn.ModuleList(
            [
                TransNeXtBlock(
                    dims[3], heads[3], drop_path=float(next(dp_iter, 0.0)), dropout=float(dropout)
                )
                for _ in range(depths[3])
            ]
        )

        self.norm = nn.LayerNorm(dims[-1])
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(dims[-1], int(num_classes))

    def _run_stage(
        self, t: torch.Tensor, hw: tuple[int, int], blocks: nn.ModuleList
    ) -> torch.Tensor:
        for b in blocks:
            t = b(t, hw=hw)
        return t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        t, hw = self.embed1(x)
        t = self._run_stage(t, hw, self.stage1)

        b, n, c = t.shape
        x2d = t.transpose(1, 2).contiguous().view(b, c, hw[0], hw[1])

        x2d = self.down2(x2d)
        hw = (int(x2d.shape[-2]), int(x2d.shape[-1]))
        t = x2d.flatten(2).transpose(1, 2).contiguous()
        t = self._run_stage(t, hw, self.stage2)

        b, n, c = t.shape
        x2d = t.transpose(1, 2).contiguous().view(b, c, hw[0], hw[1])
        x2d = self.down3(x2d)
        hw = (int(x2d.shape[-2]), int(x2d.shape[-1]))
        t = x2d.flatten(2).transpose(1, 2).contiguous()
        t = self._run_stage(t, hw, self.stage3)

        b, n, c = t.shape
        x2d = t.transpose(1, 2).contiguous().view(b, c, hw[0], hw[1])
        x2d = self.down4(x2d)
        hw = (int(x2d.shape[-2]), int(x2d.shape[-1]))
        t = x2d.flatten(2).transpose(1, 2).contiguous()
        t = self._run_stage(t, hw, self.stage4)

        t = self.norm(t)
        t = self.drop(t.mean(dim=1))
        return self.head(t)


_VARIANTS: dict[str, dict] = {
    "transnext_tiny": {
        "dims": (64, 128, 256, 384),
        "depths": (1, 1, 3, 1),
        "heads": (2, 4, 8, 12),
        "patch": 4,
    },
    "transnext_small": {
        "dims": (64, 160, 320, 512),
        "depths": (1, 2, 4, 2),
        "heads": (2, 5, 10, 16),
        "patch": 4,
    },
    "transnext_base": {
        "dims": (80, 192, 384, 640),
        "depths": (2, 2, 6, 2),
        "heads": (2, 6, 12, 20),
        "patch": 4,
    },
}


def build_transnext_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "transnext_tiny",
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown TransNeXt variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return TransNeXtClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dims=tuple(map(int, spec["dims"])),
        depths=tuple(map(int, spec["depths"])),
        heads=tuple(map(int, spec["heads"])),
        patch_size=int(spec["patch"]),
        width_mult=float(width_mult),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_transnext_classifier(
        in_channels=3, num_classes=10, variant="transnext_tiny", width_mult=0.5
    )
    y = m(x)
    print("transnext_tiny", tuple(y.shape))
