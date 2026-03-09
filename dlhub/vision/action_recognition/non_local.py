"""Non-local block - toy-first video action classifier.

Reference:
- "Non-local Neural Networks" (CVPR 2018)

Toy interpretation:
- Insert a light non-local self-attention block into a small 3D CNN.
- Important: apply the non-local block after spatial/temporal downsampling to keep
  attention matrix sizes manageable on CPU.
"""

import math

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


class NonLocal3D(nn.Module):
    """Embedded-Gaussian non-local block on (B,C,T,H,W).

    This is a simplified version intended for small feature maps.
    """

    def __init__(self, channels: int, *, reduction: int = 2) -> None:
        super().__init__()
        c = int(channels)
        r = max(1, int(reduction))
        inter = max(8, c // r)

        self.theta = nn.Conv3d(c, inter, kernel_size=1, bias=False)
        self.phi = nn.Conv3d(c, inter, kernel_size=1, bias=False)
        self.g = nn.Conv3d(c, inter, kernel_size=1, bias=False)
        self.out = nn.Conv3d(inter, c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(c)

        self.scale = 1.0 / math.sqrt(float(inter))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        thw = int(t * h * w)
        # Guard: keep this toy block sane.
        if thw > 1024:
            raise ValueError(
                f"NonLocal3D expects a downsampled feature map (T*H*W <= 1024), got {t}*{h}*{w}={thw}."
            )

        theta = self.theta(x).view(b, -1, thw).transpose(1, 2)  # (B, THW, C')
        phi = self.phi(x).view(b, -1, thw)  # (B, C', THW)
        attn = torch.softmax((theta @ phi) * float(self.scale), dim=-1)  # (B, THW, THW)

        g = self.g(x).view(b, -1, thw).transpose(1, 2)  # (B, THW, C')
        y = attn @ g  # (B, THW, C')
        y = y.transpose(1, 2).contiguous().view(b, -1, t, h, w)  # (B, C', T, H, W)
        y = self.bn(self.out(y))
        return x + y


class NonLocalVideoClassifier(nn.Module):
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

        # Downsample first to keep the non-local attention cheap.
        self.stem = nn.Sequential(
            Conv3dBNAct(c_in, w, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            Conv3dBNAct(w, 2 * w, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )

        # Attention happens here on a small (T,H,W) grid.
        self.non_local = NonLocal3D(2 * w, reduction=2)

        tail: list[nn.Module] = []
        ch = 2 * w
        for i in range(d):
            out = ch if i == 0 else ch * 2
            tail.append(Conv3dBNAct(ch, out, kernel_size=3, stride=1, padding=1))
            ch = out
            if i % 2 == 0:
                tail.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))

        self.tail = nn.Sequential(*tail)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(ch), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_video_input(x)
        x = self.stem(x)
        x = self.non_local(x)
        x = self.tail(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1)).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


_VARIANTS: dict[str, dict] = {
    "non_local_tiny": {"width": 16, "depth": 1},
    "non_local_small": {"width": 24, "depth": 2},
    "non_local_base": {"width": 32, "depth": 3},
}


def build_non_local_video_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "non_local_small",
    width_mult: float = 1.0,
    dropout: float = 0.1,
    # Unused but kept for zoo/script signature consistency.
    frames: int = 8,
    image_size: int = 64,
) -> nn.Module:
    del frames, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Non-local variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    width = scale_channels(int(spec["width"]), float(width_mult), min_ch=8, divisor=8)
    return NonLocalVideoClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=int(width),
        depth=int(spec["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 64, 64)
    m = build_non_local_video_classifier(in_channels=3, num_classes=6, variant="non_local_tiny", width_mult=0.5, dropout=0.0)
    y = m(x)
    print("non_local_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")

