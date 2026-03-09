"""VideoRNN (CNN + GRU) - toy-first video action classifier.

Reference (trend):
- "Bringing RNNs Back to Efficient Open-Ended Video Understanding" (ICCV 2025)

Toy interpretation:
- Per-frame 2D CNN -> sequence of frame embeddings -> GRU -> classify.
- This is intentionally tiny and dependency-free (no flow, no pretrained weights).
"""

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels

from ._common import check_video_input


class TinyFrameCNN(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) -> (B,F)
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected frame input shape (B, C, H, W), got {tuple(x.shape)}")
        x = self.net(x)
        return F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)


class VideoRNNVideoClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        depth: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        w = int(width)
        d = int(depth)
        h = int(hidden_dim)
        n = int(layers)
        if n <= 0:
            raise ValueError("layers must be > 0")
        if h <= 0:
            raise ValueError("hidden_dim must be > 0")

        self.backbone = TinyFrameCNN(in_channels=int(in_channels), width=w, depth=d)
        self.rnn = nn.GRU(
            input_size=int(self.backbone.out_dim),
            hidden_size=h,
            num_layers=n,
            batch_first=True,
            dropout=float(dropout) if n > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(h, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_video_input(x)  # (B,C,T,H,W)
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        feats = self.backbone(frames).view(b, t, -1)  # (B,T,F)
        out, hn = self.rnn(feats)
        last = hn[-1]  # (B,H)
        last = self.dropout(last)
        return self.classifier(last)


_VARIANTS: dict[str, dict] = {
    "videornn_tiny": {"width": 24, "depth": 2, "hidden": 128, "layers": 1},
    "videornn_small": {"width": 32, "depth": 3, "hidden": 160, "layers": 1},
    "videornn_base": {"width": 48, "depth": 4, "hidden": 192, "layers": 2},
}


def build_videornn_video_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "videornn_small",
    width_mult: float = 1.0,
    dropout: float = 0.1,
    # Unused but kept for zoo/script signature consistency.
    frames: int = 8,
    image_size: int = 64,
) -> nn.Module:
    del frames, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown VideoRNN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    width = scale_channels(int(spec["width"]), float(width_mult), min_ch=8, divisor=8)
    hidden = scale_channels(int(spec["hidden"]), float(width_mult), min_ch=32, divisor=8)
    return VideoRNNVideoClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=int(width),
        depth=int(spec["depth"]),
        hidden_dim=int(hidden),
        layers=int(spec["layers"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 64, 64)
    m = build_videornn_video_classifier(in_channels=3, num_classes=6, variant="videornn_tiny", width_mult=0.5, dropout=0.0)
    y = m(x)
    print("videornn_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")

