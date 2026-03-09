
import torch
from torch import nn


class AllConvNetClassifier(nn.Module):
    """All Convolutional Net (All-CNN-C style, simplified).

    - No fully-connected layers (except final classifier conv)
    - Strided convs instead of pooling
    - Global average pooling for logits
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        channels: tuple[int, int] = (96, 192),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        c1, c2 = int(channels[0]), int(channels[1])

        def conv(in_ch: int, out_ch: int, *, k: int = 3, s: int = 1, p: int | None = None) -> nn.Conv2d:
            if p is None:
                p = k // 2
            return nn.Conv2d(int(in_ch), int(out_ch), kernel_size=int(k), stride=int(s), padding=int(p), bias=True)

        self.features = nn.Sequential(
            conv(int(in_channels), c1, k=3, s=1),
            nn.ReLU(inplace=True),
            conv(c1, c1, k=3, s=1),
            nn.ReLU(inplace=True),
            conv(c1, c1, k=3, s=2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            conv(c1, c2, k=3, s=1),
            nn.ReLU(inplace=True),
            conv(c2, c2, k=3, s=1),
            nn.ReLU(inplace=True),
            conv(c2, c2, k=3, s=2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            conv(c2, c2, k=3, s=1),
            nn.ReLU(inplace=True),
            conv(c2, c2, k=1, s=1, p=0),
            nn.ReLU(inplace=True),
            conv(c2, int(num_classes), k=1, s=1, p=0),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.features(x)
        x = self.pool(x)
        return self.flatten(x)


_VARIANTS: dict[str, dict] = {
    "all_convnet_small": {"channels": (64, 128), "dropout": 0.2},
    "all_convnet_base": {"channels": (96, 192), "dropout": 0.2},
    "all_convnet_large": {"channels": (128, 256), "dropout": 0.3},
}


def build_all_convnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "all_convnet_base",
    dropout: float | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown AllConvNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return AllConvNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        channels=tuple(map(int, spec["channels"])),
        dropout=float(spec["dropout"] if dropout is None else dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["all_convnet_small", "all_convnet_base"]:
        m = build_all_convnet_classifier(in_channels=3, num_classes=10, variant=v)
        y = m(x)
        print(v, tuple(y.shape))

