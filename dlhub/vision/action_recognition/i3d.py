"""I3D (Inflated 3D ConvNet) - toy-first video action classifier.

Reference:
- "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset" (CVPR 2017)

Toy interpretation:
- Use a tiny 3D CNN with Inception-style multi-branch blocks.
- Keep the goal educational: show "inflate to 3D" + multi-branch mixing.
"""

import torch
import torch.nn.functional as F
from torch import nn

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
    ) -> None:
        if padding is None:
            if isinstance(kernel_size, tuple):
                padding = tuple(int(k) // 2 for k in kernel_size)
            else:
                padding = int(kernel_size) // 2

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
            nn.ReLU(inplace=True),
        )


class Inception3DBlock(nn.Module):
    """A very small Inception-style block for 3D features."""

    def __init__(self, *, in_channels: int, branch_width: int) -> None:
        super().__init__()
        c_in = int(in_channels)
        w = int(branch_width)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if w <= 0:
            raise ValueError("branch_width must be > 0")

        # Keep every branch the same width; concatenate.
        self.b1 = Conv3dBNAct(c_in, w, kernel_size=1, stride=1, padding=0)

        self.b2 = nn.Sequential(
            Conv3dBNAct(c_in, w, kernel_size=1, stride=1, padding=0),
            Conv3dBNAct(w, w, kernel_size=3, stride=1, padding=1),
        )

        # Use another 3x3x3 instead of 5x5x5 to keep compute low.
        self.b3 = nn.Sequential(
            Conv3dBNAct(c_in, w, kernel_size=1, stride=1, padding=0),
            Conv3dBNAct(w, w, kernel_size=3, stride=1, padding=1),
        )

        self.b4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            Conv3dBNAct(c_in, w, kernel_size=1, stride=1, padding=0),
        )

        self.out_channels = 4 * int(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B, C, T, H, W), got {tuple(x.shape)}")
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return y


class I3DVideoClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        w = int(width)
        d = int(depth)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if w < 8:
            raise ValueError("width must be >= 8")
        if d <= 0:
            raise ValueError("depth must be > 0")

        self.stem = nn.Sequential(
            Conv3dBNAct(c_in, w, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        blocks: list[nn.Module] = []
        ch = w
        for i in range(d):
            bw = ch if i == 0 else min(4 * w, ch * 2)
            inc = Inception3DBlock(in_channels=ch, branch_width=int(bw))
            blocks.append(inc)
            ch = int(inc.out_channels)
            if i % 2 == 0:
                blocks.append(nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))

        self.blocks = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(ch), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_video_input(x)
        x = self.stem(x)
        x = self.blocks(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1)).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


_VARIANTS: dict[str, dict] = {
    "i3d_tiny": {"width": 12, "depth": 2},
    "i3d_small": {"width": 16, "depth": 3},
    "i3d_base": {"width": 24, "depth": 4},
}


def build_i3d_video_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "i3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.1,
    # Unused but kept for zoo/script signature consistency.
    frames: int = 8,
    image_size: int = 64,
) -> nn.Module:
    del frames, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown I3D variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    width = scale_channels(int(spec["width"]), float(width_mult), min_ch=8, divisor=4)
    return I3DVideoClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=int(width),
        depth=int(spec["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 64, 64)
    m = build_i3d_video_classifier(
        in_channels=3, num_classes=6, variant="i3d_tiny", width_mult=0.75, dropout=0.0
    )
    y = m(x)
    print("i3d_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
