
import torch
from torch import nn


class ConvMixerBlock(nn.Module):
    def __init__(self, dim: int, *, kernel_size: int, dropout: float) -> None:
        super().__init__()
        k = int(kernel_size)
        d = int(dim)
        self.dw = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=k, padding=k // 2, groups=d),
            nn.GELU(),
            nn.BatchNorm2d(d),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(d),
        )
        self.drop = nn.Dropout2d(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.dw(x))
        x = self.pw(x)
        return x


class ConvMixerClassifier(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        kernel_size: int = 9,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if int(image_size) % int(patch_size) != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.patch_embed = nn.Sequential(
            nn.Conv2d(int(in_channels), int(embed_dim), kernel_size=int(patch_size), stride=int(patch_size)),
            nn.GELU(),
            nn.BatchNorm2d(int(embed_dim)),
        )
        self.blocks = nn.Sequential(
            *[
                ConvMixerBlock(int(embed_dim), kernel_size=int(kernel_size), dropout=float(dropout))
                for _ in range(int(depth))
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(int(embed_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


_SPECS: dict[str, dict] = {
    "tiny": {"embed_dim": 128, "depth": 4},
    "small": {"embed_dim": 192, "depth": 8},
    "base": {"embed_dim": 256, "depth": 12},
}


def build_convmixer_classifier(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    patch_size: int = 8,
    variant: str = "tiny",
    kernel_size: int = 9,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _SPECS:
        raise ValueError(f"Unknown ConvMixer variant: {variant!r}. Supported: {sorted(_SPECS)}")
    spec = _SPECS[name]
    return ConvMixerClassifier(
        image_size=int(image_size),
        patch_size=int(patch_size),
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        embed_dim=int(spec["embed_dim"]),
        depth=int(spec["depth"]),
        kernel_size=int(kernel_size),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["tiny", "small", "base"]:
        m = build_convmixer_classifier(in_channels=3, num_classes=10, variant=v, patch_size=8)
        y = m(x)
        print(f"convmixer_{v}", tuple(y.shape))

