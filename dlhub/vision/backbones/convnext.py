
import torch
from torch import nn

from dlhub.vision.backbones._blocks import LayerNorm2d, scale_channels


class ConvNeXtBlock(nn.Module):
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
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.ln(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        if self.gamma is not None:
            x = x * self.gamma
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return identity + x


class ConvNeXtClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dims: tuple[int, int, int, int],
        depths: tuple[int, int, int, int],
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        w = float(width_mult)
        dims = tuple(scale_channels(int(d), w, min_ch=16, divisor=8) for d in dims)
        depths = tuple(map(int, depths))

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(int(in_channels), dims[0], kernel_size=4, stride=4),
                LayerNorm2d(dims[0], eps=1e-6),
            )
        )
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(dims[i], eps=1e-6),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )

        self.stages = nn.ModuleList([nn.Sequential(*[ConvNeXtBlock(dims[i]) for _ in range(depths[i])]) for i in range(4)])

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(dims[-1], int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean(dim=(2, 3))
        x = self.norm(x)
        x = self.drop(x)
        return self.head(x)


def build_convnext_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "convnext_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name == "convnext_tiny":
        dims = (96, 192, 384, 768)
        depths = (3, 3, 9, 3)
    elif name == "convnext_small":
        dims = (96, 192, 384, 768)
        depths = (3, 3, 27, 3)
    elif name == "convnext_base":
        dims = (128, 256, 512, 1024)
        depths = (3, 3, 27, 3)
    elif name == "convnext_large":
        dims = (192, 384, 768, 1536)
        depths = (3, 3, 27, 3)
    else:
        raise ValueError("Unknown ConvNeXt variant. Supported: convnext_tiny|convnext_small|convnext_base|convnext_large")

    return ConvNeXtClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dims=tuple(map(int, dims)),
        depths=tuple(map(int, depths)),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["convnext_tiny", "convnext_small"]:
        m = build_convnext_classifier(in_channels=3, num_classes=10, variant=v, width_mult=0.5)
        y = m(x)
        print(v, tuple(y.shape))

