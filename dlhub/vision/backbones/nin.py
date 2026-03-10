import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


class MLPConv(nn.Module):
    """NiN MLPConv: conv + 1x1 + 1x1."""

    def __init__(
        self, in_ch: int, out_ch: int, *, kernel_size: int, stride: int, padding: int
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(
                in_ch,
                out_ch,
                kernel_size=int(kernel_size),
                stride=int(stride),
                padding=int(padding),
                act="relu",
            ),
            ConvBNAct(out_ch, out_ch, kernel_size=1, stride=1, padding=0, act="relu"),
            ConvBNAct(out_ch, out_ch, kernel_size=1, stride=1, padding=0, act="relu"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NiNClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        variant: str = "nin",
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        name = str(variant).lower().strip()
        if name not in {"nin", "nin_wide"}:
            raise ValueError("Unknown NiN variant. Supported: nin, nin_wide")

        w = float(width_mult) * (1.25 if name == "nin_wide" else 1.0)
        c1 = scale_channels(192, w, min_ch=32, divisor=8)
        c2 = scale_channels(256, w, min_ch=32, divisor=8)
        c3 = scale_channels(384, w, min_ch=32, divisor=8)

        self.features = nn.Sequential(
            MLPConv(int(in_channels), c1, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=float(dropout)),
            MLPConv(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=float(dropout)),
            MLPConv(c2, c3, kernel_size=3, stride=1, padding=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c3, int(num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.features(x)
        return self.head(x)


def build_nin_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "nin",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return NiNClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(variant),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["nin", "nin_wide"]:
        m = build_nin_classifier(in_channels=3, num_classes=10, variant=v, width_mult=1.0)
        y = m(x)
        print(v, tuple(y.shape))
