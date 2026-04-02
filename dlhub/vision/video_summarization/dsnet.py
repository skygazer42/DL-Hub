from __future__ import annotations

import torch
from torch import nn

from ._common import SegmentPooler, TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "dsnet_tiny": {"width": 24, "depth": 2},
    "dsnet_small": {"width": 32, "depth": 3},
    "dsnet_base": {"width": 48, "depth": 4},
}


class DSNetVideoSummarizer(nn.Module):
    """DSNet-style detect-to-summarize scorer (toy)."""

    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = TinyFrameEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        hidden = max(32, int(self.encoder.out_dim) // 2)
        self.pooler = SegmentPooler(
            dim=int(self.encoder.out_dim),
            hidden_dim=hidden,
            dropout=float(dropout),
        )
        self.windows = (2, 4, 6)

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        raw_scores, seg_scores = self.pooler(feat, windows=self.windows)
        scores = torch.sigmoid(raw_scores)
        summary_mask = scores_to_mask(scores)
        return {"scores": scores, "summary_mask": summary_mask, "segment_scores": seg_scores}


def build_dsnet_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "dsnet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DSNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return DSNetVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_dsnet_video_summarizer(in_channels=3, variant="dsnet_tiny", width_mult=0.5)
    out = m(x)
    print("dsnet_tiny", tuple(out["scores"].shape), tuple(out["segment_scores"].shape))
    loss = out["scores"].mean() + out["segment_scores"].mean()
    loss.backward()
    print("ok")
