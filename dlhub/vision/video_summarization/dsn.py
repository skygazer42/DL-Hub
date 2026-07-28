from __future__ import annotations

import torch
from torch import nn

from ._common import TinyFrameEncoder, TemporalGRUScorer, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "dsn_tiny": {"width": 24, "depth": 2},
    "dsn_small": {"width": 32, "depth": 3},
    "dsn_base": {"width": 48, "depth": 4},
}


class DSNVideoSummarizer(nn.Module):
    """DSN-style frame importance scorer (compact)."""

    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = TinyFrameEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        hidden = max(32, int(self.encoder.out_dim) // 2)
        self.scorer = TemporalGRUScorer(
            dim=int(self.encoder.out_dim),
            hidden_dim=hidden,
            layers=1,
            dropout=float(dropout),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        scores = torch.sigmoid(self.scorer(feat))
        summary_mask = scores_to_mask(scores)
        return {"scores": scores, "summary_mask": summary_mask, "features": feat}


def build_dsn_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "dsn_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DSN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return DSNVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_dsn_video_summarizer(in_channels=3, variant="dsn_tiny", width_mult=0.5)
    out = m(x)
    print("dsn_tiny", tuple(out["scores"].shape), tuple(out["summary_mask"].shape))
    loss = out["scores"].mean() + out["features"].mean()
    loss.backward()
    print("ok")
