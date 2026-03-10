import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead


class SpatialTransformer(nn.Module):
    """Spatial Transformer Network (STN) module.

    Predicts a 2x3 affine matrix and warps the input via grid_sample.
    """

    def __init__(self, in_channels: int, *, hidden: int = 32) -> None:
        super().__init__()
        h = int(hidden)
        self.loc = nn.Sequential(
            ConvBNAct(int(in_channels), h, kernel_size=7, stride=2, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBNAct(h, 2 * h, kernel_size=5, stride=2, act="relu"),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(2 * h * 4 * 4, 128),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(128, 6)
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        # identity init
        with torch.no_grad():
            self.fc.bias.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=self.fc.bias.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.loc(x).to(x.dtype)
        theta = self.fc(theta).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


class STNNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        use_stn: bool = True,
        width: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        w = int(width)
        self.stn = (
            SpatialTransformer(int(in_channels), hidden=32) if bool(use_stn) else nn.Identity()
        )
        self.backbone = nn.Sequential(
            ConvBNAct(int(in_channels), w, kernel_size=3, stride=2, act="relu"),
            ConvBNAct(w, w, kernel_size=3, stride=1, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBNAct(w, 2 * w, kernel_size=3, stride=2, act="relu"),
            ConvBNAct(2 * w, 2 * w, kernel_size=3, stride=1, act="relu"),
            ConvBNAct(2 * w, 4 * w, kernel_size=3, stride=2, act="relu"),
        )
        self.head = GlobalAvgPoolHead(4 * w, int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stn(x)
        x = self.backbone(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "stn_net_on": {"use_stn": True, "width": 64},
    "stn_net_off": {"use_stn": False, "width": 64},
    "stn_net_tiny": {"use_stn": True, "width": 48},
}


def build_stn_net_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "stn_net_on",
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown STNNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return STNNetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        use_stn=bool(spec["use_stn"]),
        width=int(spec["width"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["stn_net_on", "stn_net_off"]:
        m = build_stn_net_classifier(in_channels=3, num_classes=10, variant=v)
        y = m(x)
        print(v, tuple(y.shape))
