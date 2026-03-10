import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct


class SpatialPyramidPooling(nn.Module):
    def __init__(self, levels: tuple[int, ...] = (1, 2, 4), *, mode: str = "max") -> None:
        super().__init__()
        lvls = tuple(int(level) for level in levels)
        if not lvls or any(level <= 0 for level in lvls):
            raise ValueError("levels must be a non-empty tuple of positive ints")
        self.levels = lvls
        self.mode = str(mode).lower().strip()
        if self.mode not in {"max", "avg"}:
            raise ValueError("mode must be 'max' or 'avg'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        feats: list[torch.Tensor] = []
        for level in self.levels:
            if self.mode == "max":
                y = F.adaptive_max_pool2d(x, output_size=(level, level))
            else:
                y = F.adaptive_avg_pool2d(x, output_size=(level, level))
            feats.append(y.reshape(b, c * level * level))
        return torch.cat(feats, dim=1)


class SPPNetClassifier(nn.Module):
    """SPPNet (Spatial Pyramid Pooling) style classifier (simplified)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        base_channels: int = 64,
        levels: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        c = int(base_channels)
        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), c, kernel_size=7, stride=2, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.features = nn.Sequential(
            ConvBNAct(c, 2 * c, kernel_size=3, stride=1, act="relu"),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNAct(2 * c, 4 * c, kernel_size=3, stride=1, act="relu"),
        )
        self.spp = SpatialPyramidPooling(tuple(levels), mode="max")
        self.drop = nn.Dropout(p=float(dropout))

        spp_dim = (4 * c) * sum(int(level) * int(level) for level in levels)
        self.fc = nn.Linear(int(spp_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.features(x)
        x = self.spp(x)
        x = self.drop(x)
        return self.fc(x)


_VARIANTS: dict[str, dict] = {
    "sppnet_tiny": {"base_channels": 32, "levels": (1, 2, 4)},
    "sppnet_base": {"base_channels": 64, "levels": (1, 2, 4)},
    "sppnet_wide": {"base_channels": 96, "levels": (1, 2, 4)},
    "sppnet_fine": {"base_channels": 64, "levels": (1, 2, 4, 6)},
}


def build_sppnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "sppnet_base",
    dropout: float = 0.2,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SPPNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return SPPNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        base_channels=int(spec["base_channels"]),
        levels=tuple(map(int, spec["levels"])),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["sppnet_tiny", "sppnet_base", "sppnet_fine"]:
        m = build_sppnet_classifier(in_channels=3, num_classes=10, variant=v)
        y = m(x)
        print(v, tuple(y.shape))
