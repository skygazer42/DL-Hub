from __future__ import annotations

import torch
from torch import nn

from ._common import (
    SegmentPooler,
    TinyFrameEncoder,
    scores_to_mask,
)

_VARIANTS: dict[str, dict[str, int]] = {
    "longcontext_sum_tiny": {"width": 24, "depth": 2},
    "longcontext_sum_small": {"width": 32, "depth": 3},
    "longcontext_sum_base": {"width": 48, "depth": 4},
}


class LongcontextSumVideoSummarizer(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = TinyFrameEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        self.pooler = SegmentPooler(
            dim=int(self.encoder.out_dim),
            hidden_dim=max(16, int(self.encoder.out_dim)),
            dropout=float(dropout),
        )
        self.windows = (4, 8, 12)

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        frame_scores, segment_scores = self.pooler(feat, windows=self.windows)
        scores = torch.sigmoid(frame_scores)
        return {
            "scores": scores,
            "summary_mask": scores_to_mask(scores),
            "segment_scores": segment_scores,
        }


def build_longcontext_sum_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "longcontext_sum_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    cfg = _VARIANTS[str(variant).lower().strip()]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return LongcontextSumVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_longcontext_sum_video_summarizer(
        in_channels=3, variant="longcontext_sum_tiny", width_mult=0.5
    )
    out = m(x)
    print(
        "longcontext_sum_tiny",
        {k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)},
    )
    loss = sum(v.mean() for v in out.values() if isinstance(v, torch.Tensor))
    loss.backward()
    print("ok")
