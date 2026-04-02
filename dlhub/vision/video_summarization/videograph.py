from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ._common import TemporalAttentionScorer, TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "videograph_tiny": {"width": 24, "depth": 2},
    "videograph_small": {"width": 32, "depth": 3},
    "videograph_base": {"width": 48, "depth": 4},
}


class VideoGraphSummarizer(nn.Module):
    """Graph-based video summarizer with lightweight message passing."""

    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.steps = max(1, int(depth) - 1)
        self.encoder = TinyFrameEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        dim = int(self.encoder.out_dim)
        hidden = max(32, dim // 2)
        self.message_proj = nn.Linear(dim, dim)
        self.update = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, dim),
            nn.ReLU(inplace=True),
        )
        self.scorer = TemporalAttentionScorer(
            dim=dim,
            heads=4,
            depth=max(1, int(depth) - 1),
            dropout=float(dropout),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        b, t, d = feat.shape
        norm_feat = F.normalize(feat, dim=-1)
        sim = torch.matmul(norm_feat, norm_feat.transpose(1, 2))

        pos = torch.arange(int(t), device=feat.device, dtype=feat.dtype)
        dist = (pos.view(1, int(t), 1) - pos.view(1, 1, int(t))).abs()
        temporal_bias = torch.exp(-dist / max(1.0, float(int(t) - 1)))
        affinity = torch.softmax(sim + temporal_bias, dim=-1)

        h = feat
        for _ in range(int(self.steps)):
            msg = torch.matmul(affinity, self.message_proj(h))
            h = h + self.update(torch.cat([h, msg], dim=-1)) / math.sqrt(max(1, int(d)))

        raw_scores = self.scorer(h)
        degree = affinity.mean(dim=-1)
        scores = torch.sigmoid(raw_scores + degree)
        summary_mask = scores_to_mask(scores)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "graph_affinity": affinity,
            "node_features": h,
        }


def build_videograph_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "videograph_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown VideoGraph variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return VideoGraphSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_videograph_video_summarizer(
        in_channels=3,
        variant="videograph_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("videograph_tiny", tuple(out["scores"].shape), tuple(out["graph_affinity"].shape))
    loss = out["scores"].mean() + out["graph_affinity"].mean()
    loss.backward()
    print("ok")
