from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath
from dlhub.vision.backbones._transformer import MLP, PatchEmbed


class PatchConvMixer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        d = int(dim)
        self.dw = nn.Conv2d(d, d, kernel_size=3, padding=1, groups=d, bias=False)
        self.pw = nn.Conv2d(d, d, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(d)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)


class PatchConvNetBlock(nn.Module):
    def __init__(self, dim: int, *, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = nn.LayerNorm(d)
        self.mix = PatchConvMixer(d)
        self.dp1 = DropPath(float(drop_path))
        self.norm2 = nn.LayerNorm(d)
        self.mlp = MLP(d, 4 * d, dropout=0.0, act="gelu")
        self.dp2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor, *, hw: tuple[int, int]) -> torch.Tensor:
        b, n, d = x.shape
        h, w = int(hw[0]), int(hw[1])
        y = self.norm1(x).view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        y = self.mix(y).flatten(2).transpose(1, 2)
        x = x + self.dp1(y)
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


class PatchConvNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int = 64,
        patch_size: int = 4,
        dim: int = 192,
        depth: int = 8,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch = PatchEmbed(int(in_channels), int(dim), patch_size=int(patch_size))
        h = int(image_size) // int(patch_size)
        w = int(image_size) // int(patch_size)
        self.hw = (h, w)
        self.pos = nn.Parameter(torch.zeros(1, h * w, int(dim)))
        dp_rates = torch.linspace(0.0, float(drop_path), steps=int(depth)).tolist()
        self.blocks = nn.ModuleList([PatchConvNetBlock(int(dim), drop_path=float(dp_rates[i])) for i in range(int(depth))])
        self.norm = nn.LayerNorm(int(dim))
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(int(dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch(x) + self.pos
        for b in self.blocks:
            x = b(x, hw=self.hw)
        x = self.norm(x)
        x = self.drop(x.mean(dim=1))
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "patchconvnet_tiny": {"dim": 192, "depth": 8, "patch": 4},
    "patchconvnet_small": {"dim": 256, "depth": 10, "patch": 4},
}


def build_patchconvnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "patchconvnet_tiny",
    image_size: int = 64,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown PatchConvNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return PatchConvNetClassifier(
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
    m = build_patchconvnet_classifier(in_channels=3, num_classes=10, variant="patchconvnet_tiny", image_size=64)
    y = m(x)
    print("patchconvnet_tiny", tuple(y.shape))

