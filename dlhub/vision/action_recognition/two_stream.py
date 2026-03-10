"""Two-Stream CNN (RGB + motion) - toy-first video action classifier.

Reference (classic idea):
- "Two-Stream Convolutional Networks for Action Recognition in Videos" (NIPS 2014)

Toy interpretation:
- Keep a single NCTHW input tensor (no optical flow downloads).
- RGB stream: per-frame 2D CNN + temporal mean pooling.
- Motion stream: simple frame-difference "pseudo-flow" + per-frame 2D CNN + temporal mean pooling.
- Fuse the two pooled embeddings then classify.
"""

import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels

from ._common import check_video_input


def _frame_diff_motion(x: torch.Tensor) -> torch.Tensor:
    """Make a cheap motion surrogate from frames: x[t] - x[t-1]."""

    if x.ndim != 5:
        raise ValueError(f"Expected video input shape (B, C, T, H, W), got {tuple(x.shape)}")
    if x.shape[2] <= 1:
        return torch.zeros_like(x)

    d = x[:, :, 1:] - x[:, :, :-1]
    z = torch.zeros_like(x[:, :, :1])
    return torch.cat([z, d], dim=2)


class TinyFrameCNN(nn.Module):
    """A tiny 2D CNN that maps a frame (B,C,H,W) -> (B,F)."""

    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float) -> None:
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

        layers: list[nn.Module] = [
            ConvBNAct(c_in, w, kernel_size=3, stride=2, act="relu"),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        ch = w
        for i in range(d):
            out = ch if i == 0 else ch * 2
            layers.append(ConvBNAct(ch, out, kernel_size=3, stride=1, act="relu"))
            ch = out
            if i % 2 == 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.net = nn.Sequential(*layers)
        self.out_dim = int(ch)
        self.dropout = nn.Dropout(float(dropout))
        self.proj = nn.Linear(int(ch), int(ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected frame input shape (B, C, H, W), got {tuple(x.shape)}")
        x = self.net(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = self.dropout(x)
        return torch.tanh(self.proj(x))


class TwoStreamVideoClassifier(nn.Module):
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
        self.rgb = TinyFrameCNN(
            in_channels=int(in_channels), width=int(width), depth=int(depth), dropout=float(dropout)
        )
        self.motion = TinyFrameCNN(
            in_channels=int(in_channels), width=int(width), depth=int(depth), dropout=float(dropout)
        )
        fused = int(self.rgb.out_dim + self.motion.out_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(fused, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_video_input(x)  # (B,C,T,H,W)
        b, c, t, h, w = x.shape

        rgb = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        mot = _frame_diff_motion(x).permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        rgb_feat = self.rgb(rgb).view(b, t, -1).mean(dim=1)
        mot_feat = self.motion(mot).view(b, t, -1).mean(dim=1)

        fused = torch.cat([rgb_feat, mot_feat], dim=1)
        fused = self.dropout(fused)
        return self.classifier(fused)


_VARIANTS: dict[str, dict] = {
    "two_stream_tiny": {"width": 24, "depth": 2},
    "two_stream_small": {"width": 32, "depth": 3},
    "two_stream_base": {"width": 48, "depth": 4},
}


def build_two_stream_video_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "two_stream_small",
    width_mult: float = 1.0,
    dropout: float = 0.1,
    # Unused but kept for zoo/script signature consistency.
    frames: int = 8,
    image_size: int = 64,
) -> nn.Module:
    del frames, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Two-Stream variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    width = scale_channels(int(spec["width"]), float(width_mult), min_ch=8, divisor=8)
    return TwoStreamVideoClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=int(width),
        depth=int(spec["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 64, 64)
    m = build_two_stream_video_classifier(
        in_channels=3, num_classes=6, variant="two_stream_tiny", width_mult=0.5, dropout=0.0
    )
    y = m(x)
    print("two_stream_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")
