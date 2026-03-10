import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels


class AlexNetClassifier(nn.Module):
    """AlexNet-style classifier (adapted for 64x64 inputs).

    This stays close to the classic conv/pool stack, but uses global pooling to
    avoid hard-coding feature map sizes.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        variant: str = "alexnet",
        width_mult: float = 1.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        name = str(variant).lower().strip()
        if name not in {"alexnet", "alexnet_wide"}:
            raise ValueError("Unknown AlexNet variant. Supported: alexnet, alexnet_wide")

        w = float(width_mult) * (1.25 if name == "alexnet_wide" else 1.0)
        c1 = scale_channels(64, w, min_ch=16, divisor=8)
        c2 = scale_channels(192, w, min_ch=32, divisor=8)
        c3 = scale_channels(384, w, min_ch=32, divisor=8)
        c4 = scale_channels(256, w, min_ch=32, divisor=8)
        c5 = scale_channels(256, w, min_ch=32, divisor=8)

        self.features = nn.Sequential(
            nn.Conv2d(int(in_channels), c1, kernel_size=11, stride=2, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(c1, c2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(c5, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.features(x)
        return self.head(x)


def build_alexnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "alexnet",
    width_mult: float = 1.0,
    dropout: float = 0.2,
) -> nn.Module:
    return AlexNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["alexnet", "alexnet_wide"]:
        m = build_alexnet_classifier(in_channels=3, num_classes=10, variant=v, width_mult=1.0)
        y = m(x)
        print(v, tuple(y.shape))
