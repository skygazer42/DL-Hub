import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels


class LeNet5Classifier(nn.Module):
    """LeNet-5 style classifier (small-image friendly).

    Variants are exposed via `variant` + `width_mult`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        variant: str = "lenet5",
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        name = str(variant).lower().strip()
        if name not in {"lenet5", "lenet5_wide"}:
            raise ValueError("Unknown LeNet variant. Supported: lenet5, lenet5_wide")

        base_c1, base_c2 = (16, 32) if name == "lenet5" else (24, 48)
        c1 = scale_channels(base_c1, float(width_mult), min_ch=8, divisor=4)
        c2 = scale_channels(base_c2, float(width_mult), min_ch=8, divisor=4)

        self.features = nn.Sequential(
            nn.Conv2d(int(in_channels), c1, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c1, c2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(c2, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.features(x)
        return self.head(x)


def build_lenet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "lenet5",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return LeNet5Classifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 1, 64, 64)
    for v in ["lenet5", "lenet5_wide"]:
        m = build_lenet_classifier(in_channels=1, num_classes=10, variant=v, width_mult=1.0)
        y = m(x)
        print(v, tuple(y.shape))
