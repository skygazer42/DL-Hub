import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath, make_divisible


class AdditiveTokenMixer(nn.Module):
    """SwiftFormer additive attention (very simplified)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        d = int(dim)
        self.q = nn.Linear(d, d, bias=True)
        self.k = nn.Linear(d, d, bias=True)
        self.v = nn.Linear(d, d, bias=True)
        self.proj = nn.Linear(d, d, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        q = self.q(x)
        k = self.k(x).mean(dim=1, keepdim=True)  # global context
        v = self.v(x)
        gate = torch.sigmoid(k)
        y = q * gate + v
        return self.proj(y)


class SwiftFormerBlock(nn.Module):
    def __init__(self, dim: int, *, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = nn.LayerNorm(d)
        self.mix = AdditiveTokenMixer(d)
        self.dp1 = DropPath(float(drop_path))
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.dp2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.mix(self.norm1(x)))
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


class SwiftFormerClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int = 64,
        patch_size: int = 4,
        dim: int = 192,
        depth: int = 8,
        width_mult: float = 1.0,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        d = make_divisible(int(round(int(dim) * float(width_mult))), 8)
        self.patch = nn.Conv2d(
            int(in_channels), d, kernel_size=self.patch_size, stride=self.patch_size, bias=True
        )
        h = self.image_size // self.patch_size
        w = self.image_size // self.patch_size
        self.pos = nn.Parameter(torch.zeros(1, h * w, d))
        dp_rates = torch.linspace(0.0, float(drop_path), steps=int(depth)).tolist()
        self.blocks = nn.Sequential(
            *[SwiftFormerBlock(d, drop_path=float(dp_rates[i])) for i in range(int(depth))]
        )
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(d, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = self.blocks(x + self.pos)
        x = self.norm(x)
        x = self.drop(x.mean(dim=1))
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "swiftformer_xs": {"dim": 160, "depth": 6, "patch": 4},
    "swiftformer_s": {"dim": 192, "depth": 8, "patch": 4},
    "swiftformer_m": {"dim": 256, "depth": 10, "patch": 4},
}


def build_swiftformer_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "swiftformer_s",
    image_size: int = 64,
    width_mult: float = 1.0,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown SwiftFormer variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    return SwiftFormerClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        patch_size=int(spec["patch"]),
        dim=int(spec["dim"]),
        depth=int(spec["depth"]),
        width_mult=float(width_mult),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_swiftformer_classifier(
        in_channels=3, num_classes=10, variant="swiftformer_s", image_size=64, width_mult=0.5
    )
    y = m(x)
    print("swiftformer_s", tuple(y.shape))
