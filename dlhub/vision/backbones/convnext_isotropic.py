
import torch
from torch import nn

from dlhub.vision.backbones._blocks import LayerNorm2d, scale_channels


class ConvNeXtIsoBlock(nn.Module):
    def __init__(self, dim: int, *, layer_scale_init: float = 1e-6) -> None:
        super().__init__()
        d = int(dim)
        self.dwconv = nn.Conv2d(d, d, kernel_size=7, padding=3, groups=d)
        self.ln = nn.LayerNorm(d, eps=1e-6)
        self.pw1 = nn.Linear(d, 4 * d)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * d, d)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(d)) if float(layer_scale_init) > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = x.permute(0, 3, 1, 2)
        return identity + x


class ConvNeXtIsotropicClassifier(nn.Module):
    """ConvNeXt isotropic variant: one embedding dim, no hierarchical stages (simplified)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dim: int = 384,
        depth: int = 12,
        patch_size: int = 16,
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d = scale_channels(int(dim), float(width_mult), min_ch=16, divisor=8)
        p = int(patch_size)
        self.patch = nn.Sequential(nn.Conv2d(int(in_channels), d, kernel_size=p, stride=p), LayerNorm2d(d, eps=1e-6))
        self.blocks = nn.Sequential(*[ConvNeXtIsoBlock(d) for _ in range(int(depth))])
        self.norm = nn.LayerNorm(d, eps=1e-6)
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(d, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch(x)
        x = self.blocks(x)
        x = x.mean(dim=(2, 3))
        x = self.norm(x)
        x = self.drop(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "convnext_iso_tiny": {"dim": 256, "depth": 10, "patch": 16},
    "convnext_iso_base": {"dim": 384, "depth": 12, "patch": 16},
    "convnext_iso_large": {"dim": 512, "depth": 18, "patch": 16},
}


def build_convnext_isotropic_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "convnext_iso_base",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown ConvNeXt-Isotropic variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    return ConvNeXtIsotropicClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dim=int(spec["dim"]),
        depth=int(spec["depth"]),
        patch_size=int(spec["patch"]),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_convnext_isotropic_classifier(in_channels=3, num_classes=10, variant="convnext_iso_tiny", width_mult=0.5)
    y = m(x)
    print("convnext_iso_tiny", tuple(y.shape))

