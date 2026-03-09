
"""C3D (3D CNN) - toy-first video action classifier.

Reference (classic baseline):
- "Learning Spatiotemporal Features with 3D Convolutional Networks" (ICCV 2015)

Toy interpretation:
- A small 3D ConvNet with a few stages and global average pooling.
"""

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import scale_channels

from ._common import check_video_input


class Conv3dBNAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] | None = None,
        act: str = "relu",
    ) -> None:
        if padding is None:
            if isinstance(kernel_size, tuple):
                padding = tuple(int(k) // 2 for k in kernel_size)
            else:
                padding = int(kernel_size) // 2

        act_name = str(act).lower().strip()
        if act_name == "relu":
            act_layer: nn.Module = nn.ReLU(inplace=True)
        elif act_name == "gelu":
            act_layer = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {act!r}")

        super().__init__(
            nn.Conv3d(
                int(in_ch),
                int(out_ch),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(int(out_ch)),
            act_layer,
        )


class C3DVideoClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        w = int(width)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if w < 8:
            raise ValueError("width must be >= 8")

        self.stem = nn.Sequential(
            Conv3dBNAct(c_in, w, kernel_size=3, stride=1, padding=1, act="relu"),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        self.stage2 = nn.Sequential(
            Conv3dBNAct(w, w * 2, kernel_size=3, stride=1, padding=1, act="relu"),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )
        self.stage3 = nn.Sequential(
            Conv3dBNAct(w * 2, w * 4, kernel_size=3, stride=1, padding=1, act="relu"),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )

        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(w * 4, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_video_input(x)
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1)).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


_VARIANTS: dict[str, dict] = {
    "c3d_tiny": {"width": 24},
    "c3d_small": {"width": 32},
    "c3d_base": {"width": 48},
}


def build_c3d_video_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "c3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.1,
    # Unused but kept for zoo/script signature consistency.
    frames: int = 8,
    image_size: int = 64,
) -> nn.Module:
    del frames, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown C3D variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    width = scale_channels(int(spec["width"]), float(width_mult), min_ch=8, divisor=8)
    return C3DVideoClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=int(width),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 64, 64)
    m = build_c3d_video_classifier(in_channels=3, num_classes=6, variant="c3d_tiny", width_mult=0.5, dropout=0.0)
    y = m(x)
    print("c3d_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")

