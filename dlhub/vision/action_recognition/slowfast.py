"""SlowFast (dual-pathway) - toy-first video action classifier.

Reference:
- "SlowFast Networks for Video Recognition" (ICCV 2019)

Toy interpretation:
- Two 3D CNN pathways:
  - Slow pathway: lower frame-rate sampling, higher channel capacity.
  - Fast pathway: higher frame-rate, lower channel capacity.
- Fuse pooled features by concatenation and classify.
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


class Tiny3DPath(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        c_in = int(in_channels)
        w = int(width)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")

        layers: list[nn.Module] = [
            Conv3dBNAct(c_in, w, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        ]

        ch = w
        for i in range(d):
            out = ch if i == 0 else ch * 2
            layers.append(Conv3dBNAct(ch, out, kernel_size=3, stride=1, padding=1))
            ch = out
            if i % 2 == 0:
                layers.append(nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))

        self.net = nn.Sequential(*layers)
        self.out_dim = int(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B, C, T, H, W), got {tuple(x.shape)}")
        x = self.net(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1)).flatten(1)
        return x


class SlowFastVideoClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        slow_width: int,
        fast_width: int,
        depth: int,
        alpha: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.alpha = int(alpha)
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")

        self.slow = Tiny3DPath(
            in_channels=int(in_channels), width=int(slow_width), depth=int(depth)
        )
        self.fast = Tiny3DPath(
            in_channels=int(in_channels), width=int(fast_width), depth=max(1, int(depth) - 1)
        )

        fused = self.slow.out_dim + self.fast.out_dim
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(fused), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_video_input(x)
        slow_x = x[:, :, :: self.alpha]
        fast_x = x
        slow_feat = self.slow(slow_x)
        fast_feat = self.fast(fast_x)
        fused = torch.cat([slow_feat, fast_feat], dim=1)
        fused = self.dropout(fused)
        return self.classifier(fused)


_VARIANTS: dict[str, dict] = {
    "slowfast_tiny": {"slow_width": 24, "fast_width": 8, "depth": 2, "alpha": 4},
    "slowfast_small": {"slow_width": 32, "fast_width": 12, "depth": 3, "alpha": 4},
    "slowfast_base": {"slow_width": 48, "fast_width": 16, "depth": 4, "alpha": 4},
}


def build_slowfast_video_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "slowfast_small",
    width_mult: float = 1.0,
    dropout: float = 0.1,
    # Unused but kept for zoo/script signature consistency.
    frames: int = 8,
    image_size: int = 64,
) -> nn.Module:
    del frames, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SlowFast variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    slow_w = scale_channels(int(spec["slow_width"]), float(width_mult), min_ch=8, divisor=8)
    fast_w = scale_channels(int(spec["fast_width"]), float(width_mult), min_ch=8, divisor=8)
    return SlowFastVideoClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        slow_width=int(slow_w),
        fast_width=int(fast_w),
        depth=int(spec["depth"]),
        alpha=int(spec["alpha"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16, 64, 64)
    m = build_slowfast_video_classifier(
        in_channels=3, num_classes=6, variant="slowfast_tiny", width_mult=0.5, dropout=0.0
    )
    y = m(x)
    print("slowfast_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
