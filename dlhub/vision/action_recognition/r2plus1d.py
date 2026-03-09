"""R(2+1)D - toy-first video action classifier.

Reference:
- "A Closer Look at Spatiotemporal Convolutions for Action Recognition" (CVPR 2018)

Toy interpretation:
- Replace a 3D conv with a factorized (1,3,3) spatial conv followed by (3,1,1) temporal conv.
- Build a tiny residual stack, global average pool, then classify.
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
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] | None = None,
        groups: int = 1,
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
                groups=int(groups),
                bias=False,
            ),
            nn.BatchNorm3d(int(out_ch)),
            nn.ReLU(inplace=True),
        )


class Conv2Plus1D(nn.Module):
    """Factorized 3D conv: spatial then temporal."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        stride: tuple[int, int, int] = (1, 1, 1),
        mid_ch: int | None = None,
    ) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        st, sh, sw = map(int, stride)
        if st <= 0 or sh <= 0 or sw <= 0:
            raise ValueError("stride must be positive")

        m = int(mid_ch) if mid_ch is not None else max(8, (c_in * c_out * 3 * 3) // (c_in * 3 * 3 + 3 * c_out))
        self.spatial = Conv3dBNAct(c_in, m, kernel_size=(1, 3, 3), stride=(1, sh, sw), padding=(0, 1, 1))
        self.temporal = Conv3dBNAct(m, c_out, kernel_size=(3, 1, 1), stride=(st, 1, 1), padding=(1, 0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial(x)
        return self.temporal(x)


class R2Plus1DBlock(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int, stride: tuple[int, int, int]) -> None:
        super().__init__()
        c_in = int(in_channels)
        c_out = int(out_channels)
        st, sh, sw = map(int, stride)
        if st <= 0 or sh <= 0 or sw <= 0:
            raise ValueError("stride must be positive")

        self.conv1 = Conv2Plus1D(c_in, c_out, stride=(st, sh, sw))
        self.conv2 = Conv2Plus1D(c_out, c_out, stride=(1, 1, 1))
        if (st, sh, sw) != (1, 1, 1) or c_in != c_out:
            self.down = nn.Sequential(
                nn.Conv3d(c_in, c_out, kernel_size=1, stride=(st, sh, sw), padding=0, bias=False),
                nn.BatchNorm3d(c_out),
            )
        else:
            self.down = None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        y = self.conv1(x)
        y = self.conv2(y)
        if self.down is not None:
            identity = self.down(identity)
        return self.act(identity + y)


class R2Plus1DVideoClassifier(nn.Module):
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
            Conv2Plus1D(c_in, w, stride=(1, 2, 2)),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        # 3 stages, with increasing channels and time downsampling.
        self.stage1 = nn.Sequential(*[R2Plus1DBlock(in_channels=w, out_channels=w, stride=(1, 1, 1)) for _ in range(d)])
        self.stage2 = nn.Sequential(
            R2Plus1DBlock(in_channels=w, out_channels=2 * w, stride=(2, 2, 2)),
            *[R2Plus1DBlock(in_channels=2 * w, out_channels=2 * w, stride=(1, 1, 1)) for _ in range(max(1, d - 1))],
        )
        self.stage3 = nn.Sequential(
            R2Plus1DBlock(in_channels=2 * w, out_channels=4 * w, stride=(2, 2, 2)),
            *[R2Plus1DBlock(in_channels=4 * w, out_channels=4 * w, stride=(1, 1, 1)) for _ in range(max(1, d - 1))],
        )

        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(4 * w, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_video_input(x)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1)).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


_VARIANTS: dict[str, dict] = {
    "r2plus1d_tiny": {"width": 16, "depth": 1},
    "r2plus1d_small": {"width": 24, "depth": 2},
    "r2plus1d_base": {"width": 32, "depth": 3},
}


def build_r2plus1d_video_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "r2plus1d_small",
    width_mult: float = 1.0,
    dropout: float = 0.1,
    # Unused but kept for zoo/script signature consistency.
    frames: int = 8,
    image_size: int = 64,
) -> nn.Module:
    del frames, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown R(2+1)D variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    width = scale_channels(int(spec["width"]), float(width_mult), min_ch=8, divisor=8)
    return R2Plus1DVideoClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=int(width),
        depth=int(spec["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 64, 64)
    m = build_r2plus1d_video_classifier(in_channels=3, num_classes=6, variant="r2plus1d_tiny", width_mult=0.5, dropout=0.0)
    y = m(x)
    print("r2plus1d_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")

