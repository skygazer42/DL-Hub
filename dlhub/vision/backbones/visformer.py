from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, DropPath, GlobalAvgPoolHead, scale_channels
from dlhub.vision.backbones._transformer import TransformerEncoderBlock


def _to_tokens(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    b, c, h, w = x.shape
    t = x.flatten(2).transpose(1, 2).contiguous()
    return t, (h, w)


def _to_map(t: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    h, w = int(hw[0]), int(hw[1])
    b, n, c = t.shape
    if n != h * w:
        raise ValueError("token hw mismatch")
    return t.transpose(1, 2).contiguous().view(b, c, h, w)


class VisformerStage(nn.Module):
    def __init__(self, dim: int, *, depth: int, heads: int, drop_path: float, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        dep = int(depth)
        dp_rates = torch.linspace(0.0, float(drop_path), steps=max(1, dep)).tolist()
        self.blocks = nn.Sequential(
            *[
                TransformerEncoderBlock(d, int(heads), mlp_ratio=4.0, dropout=float(dropout), drop_path=float(dp_rates[i]))
                for i in range(dep)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, hw = _to_tokens(x)
        t = self.blocks(t)
        return _to_map(t, hw)


class VisformerClassifier(nn.Module):
    """Visformer (simplified): conv stem + conv stage + transformer stages."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int] = (64, 128, 256),
        depths: tuple[int, int, int] = (2, 2, 4),
        heads: tuple[int, int] = (4, 8),
        width_mult: float = 1.0,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dims = tuple(scale_channels(int(d), float(width_mult), min_ch=16, divisor=8) for d in dims)
        depths = tuple(int(d) for d in depths)
        heads = tuple(int(h) for h in heads)

        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), dims[0], kernel_size=3, stride=2, act="relu"),
            ConvBNAct(dims[0], dims[0], kernel_size=3, stride=2, act="relu"),
        )

        dp1, dp2 = float(drop_path) * 0.5, float(drop_path)

        self.stage1 = nn.Sequential(*[ConvBNAct(dims[0], dims[0], kernel_size=3, stride=1, act="relu") for _ in range(depths[0])])
        self.down2 = ConvBNAct(dims[0], dims[1], kernel_size=3, stride=2, act="relu")
        self.stage2 = VisformerStage(dims[1], depth=depths[1], heads=heads[0], drop_path=float(dp1), dropout=float(dropout))
        self.down3 = ConvBNAct(dims[1], dims[2], kernel_size=3, stride=2, act="relu")
        self.stage3 = VisformerStage(dims[2], depth=depths[2], heads=heads[1], drop_path=float(dp2), dropout=float(dropout))

        self.head = GlobalAvgPoolHead(dims[2], int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down2(x)
        x = self.stage2(x)
        x = self.down3(x)
        x = self.stage3(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "visformer_tiny": {"dims": (64, 128, 256), "depths": (1, 1, 2), "heads": (4, 8)},
    "visformer_small": {"dims": (64, 160, 320), "depths": (2, 2, 4), "heads": (5, 10)},
    "visformer_base": {"dims": (80, 192, 384), "depths": (2, 3, 6), "heads": (6, 12)},
}


def build_visformer_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "visformer_tiny",
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Visformer variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return VisformerClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dims=tuple(map(int, spec["dims"])),
        depths=tuple(map(int, spec["depths"])),
        heads=tuple(map(int, spec["heads"])),
        width_mult=float(width_mult),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_visformer_classifier(in_channels=3, num_classes=10, variant="visformer_tiny", width_mult=0.5)
    y = m(x)
    print("visformer_tiny", tuple(y.shape))
