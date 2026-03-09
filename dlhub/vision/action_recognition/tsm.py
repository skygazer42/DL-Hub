
"""TSM (Temporal Shift Module) - toy-first video action classifier.

Reference:
- "TSM: Temporal Shift Module for Efficient Video Understanding" (ICCV 2019)

Toy interpretation:
- Perform a cheap channel-wise temporal shift on the input tensor.
- Apply a shared 2D CNN on frames (TSN-style).
- Mean-pool across time and classify.
"""

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels

from ._common import check_video_input


def temporal_shift(x: torch.Tensor, *, fold_div: int = 8) -> torch.Tensor:
    """Temporal shift on NCTHW tensor (no parameters, cheap).

    A small portion of channels are shifted forward/backward along time.
    """

    if x.ndim != 5:
        raise ValueError(f"Expected input shape (B, C, T, H, W), got {tuple(x.shape)}")
    b, c, t, h, w = x.shape
    d = int(fold_div)
    if d <= 0:
        raise ValueError("fold_div must be > 0")
    if t <= 1:
        return x

    fold = max(1, c // d)
    out = x.clone()

    # Shift a few channels forward/backward in time.
    out[:, :fold, 1:] = x[:, :fold, :-1]
    out[:, fold : 2 * fold, :-1] = x[:, fold : 2 * fold, 1:]
    return out


class TinyFrameCNN(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float) -> None:
        super().__init__()
        c_in = int(in_channels)
        w = int(width)
        d = int(depth)
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
        self.dropout = nn.Dropout(float(dropout))
        self.proj = nn.Linear(ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected frame input shape (B, C, H, W), got {tuple(x.shape)}")
        x = self.net(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = self.dropout(x)
        return torch.tanh(self.proj(x))


class TSMVideoClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        depth: int,
        fold_div: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.fold_div = int(fold_div)
        self.backbone = TinyFrameCNN(in_channels=int(in_channels), width=int(width), depth=int(depth), dropout=float(dropout))
        feat_dim = int(width) * (2 ** max(int(depth) - 1, 0))
        self.classifier = nn.Linear(int(feat_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = check_video_input(x)
        x = temporal_shift(x, fold_div=int(self.fold_div))
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        feats = self.backbone(frames).view(b, t, -1).mean(dim=1)
        return self.classifier(feats)


_VARIANTS: dict[str, dict] = {
    "tsm_tiny": {"width": 24, "depth": 2, "fold_div": 8},
    "tsm_small": {"width": 32, "depth": 3, "fold_div": 8},
    "tsm_base": {"width": 48, "depth": 4, "fold_div": 8},
}


def build_tsm_video_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "tsm_small",
    width_mult: float = 1.0,
    dropout: float = 0.1,
    # Unused but kept for zoo/script signature consistency.
    frames: int = 8,
    image_size: int = 64,
) -> nn.Module:
    del frames, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown TSM variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    width = scale_channels(int(spec["width"]), float(width_mult), min_ch=8, divisor=8)
    return TSMVideoClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=int(width),
        depth=int(spec["depth"]),
        fold_div=int(spec["fold_div"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 64, 64)
    m = build_tsm_video_classifier(in_channels=3, num_classes=6, variant="tsm_tiny", width_mult=0.5, dropout=0.0)
    y = m(x)
    print("tsm_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")

