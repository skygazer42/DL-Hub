from __future__ import annotations

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, GlobalAvgPoolHead, scale_channels


class SequencerBlock(nn.Module):
    """Sequencer (RNN for vision) block, simplified.

    Runs LSTM along width then along height.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.lstm_w = nn.LSTM(input_size=c, hidden_size=c, num_layers=1, batch_first=True)
        self.lstm_h = nn.LSTM(input_size=c, hidden_size=c, num_layers=1, batch_first=True)
        self.proj = nn.Conv2d(c, c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # along width
        xw = x.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
        xw, _ = self.lstm_w(xw)
        xw = xw.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        # along height
        xh = xw.permute(0, 3, 2, 1).contiguous().view(b * w, h, c)
        xh, _ = self.lstm_h(xh)
        xh = xh.view(b, w, h, c).permute(0, 3, 2, 1).contiguous()
        return self.proj(xh)


class SequencerClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        dim: int = 128,
        depth: int = 6,
        width_mult: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d = scale_channels(int(dim), float(width_mult), min_ch=16, divisor=8)
        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), d, kernel_size=7, stride=2, act="relu"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.blocks = nn.Sequential(*[SequencerBlock(d) for _ in range(int(depth))])
        self.head = GlobalAvgPoolHead(d, int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "sequencer_tiny": {"dim": 96, "depth": 4},
    "sequencer_base": {"dim": 128, "depth": 6},
}


def build_sequencer_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "sequencer_base",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Sequencer variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return SequencerClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        dim=int(spec["dim"]),
        depth=int(spec["depth"]),
        width_mult=float(width_mult),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 3, 64, 64)
    m = build_sequencer_classifier(in_channels=3, num_classes=10, variant="sequencer_tiny", width_mult=0.5)
    y = m(x)
    print("sequencer_tiny", tuple(y.shape))

