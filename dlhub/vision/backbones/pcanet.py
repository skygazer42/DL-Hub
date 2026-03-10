import torch
from torch import nn


class FixedConv2d(nn.Module):
    """A simple fixed convolution layer (non-trainable weights).

    PCANet learns PCA filters; this module uses fixed orthogonal-ish random filters
    as a lightweight stand-in, keeping the rest of the pipeline torch-native.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        k = int(kernel_size)
        if padding is None:
            padding = k // 2
        w = torch.randn(int(out_ch), int(in_ch), k, k)
        w = w / (w.flatten(1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-6)
        self.weight = nn.Parameter(w, requires_grad=False)
        self.stride = int(stride)
        self.padding = int(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv2d(
            x, self.weight, bias=None, stride=self.stride, padding=self.padding
        )


class PCANetClassifier(nn.Module):
    """PCANet-inspired shallow model (fixed filters + hashing-like nonlinearity)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stage1_filters: int = 8,
        stage2_filters: int = 16,
        kernel_size: int = 5,
        pool: int = 2,
    ) -> None:
        super().__init__()
        k = int(kernel_size)
        self.conv1 = FixedConv2d(
            int(in_channels), int(stage1_filters), kernel_size=k, stride=1, padding=k // 2
        )
        self.conv2 = FixedConv2d(
            int(stage1_filters), int(stage2_filters), kernel_size=k, stride=1, padding=k // 2
        )
        self.pool = nn.AvgPool2d(kernel_size=int(pool), stride=int(pool))
        self.proj = nn.Sequential(
            nn.Conv2d(int(stage2_filters), 64, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.pool(x)
        return self.proj(x)


_VARIANTS: dict[str, dict] = {
    "pcanet_tiny": {"s1": 8, "s2": 16, "k": 5},
    "pcanet_base": {"s1": 16, "s2": 32, "k": 5},
    "pcanet_wide": {"s1": 24, "s2": 48, "k": 7},
}


def build_pcanet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "pcanet_base",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown PCANet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return PCANetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stage1_filters=int(spec["s1"]),
        stage2_filters=int(spec["s2"]),
        kernel_size=int(spec["k"]),
        pool=2,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_pcanet_classifier(in_channels=3, num_classes=10, variant="pcanet_base")
    y = m(x)
    print("pcanet_base", tuple(y.shape))
