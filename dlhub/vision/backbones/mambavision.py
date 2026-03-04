from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath
from dlhub.vision.backbones._transformer import PatchEmbed


class MambaLikeBlock(nn.Module):
    """A lightweight state-space / sequence-mixing block (Mamba-inspired, simplified).

    Uses depthwise Conv1d along the token dimension and a gated linear unit.
    """

    def __init__(self, dim: int, *, kernel_size: int = 7, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        k = int(kernel_size)
        self.norm = nn.LayerNorm(d)
        self.dw = nn.Conv1d(d, d, kernel_size=k, padding=k // 2, groups=d, bias=False)
        self.proj = nn.Linear(d, 2 * d, bias=True)
        self.out = nn.Linear(d, d, bias=True)
        self.dp = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.norm(x)
        # sequence mixing
        y = self.dw(x.transpose(1, 2)).transpose(1, 2)
        u, v = self.proj(y).chunk(2, dim=-1)
        y = u * torch.sigmoid(v)
        y = self.out(y)
        return identity + self.dp(y)


class MambaVisionClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int = 64,
        patch_size: int = 4,
        dim: int = 256,
        depth: int = 10,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch = PatchEmbed(int(in_channels), int(dim), patch_size=int(patch_size))
        h = int(image_size) // int(patch_size)
        w = int(image_size) // int(patch_size)
        self.pos = nn.Parameter(torch.zeros(1, h * w, int(dim)))
        dp_rates = torch.linspace(0.0, float(drop_path), steps=int(depth)).tolist()
        self.blocks = nn.Sequential(*[MambaLikeBlock(int(dim), kernel_size=7, drop_path=float(dp_rates[i])) for i in range(int(depth))])
        self.norm = nn.LayerNorm(int(dim))
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(int(dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch(x) + self.pos
        x = self.blocks(x)
        x = self.norm(x)
        x = self.drop(x.mean(dim=1))
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "mambavision_tiny": {"dim": 256, "depth": 10, "patch": 4},
    "mambavision_small": {"dim": 384, "depth": 12, "patch": 4},
}


def build_mambavision_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "mambavision_tiny",
    image_size: int = 64,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MambaVision variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return MambaVisionClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        patch_size=int(spec["patch"]),
        dim=int(spec["dim"]),
        depth=int(spec["depth"]),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_mambavision_classifier(in_channels=3, num_classes=10, variant="mambavision_tiny", image_size=64)
    y = m(x)
    print("mambavision_tiny", tuple(y.shape))

