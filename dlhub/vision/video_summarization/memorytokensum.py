from __future__ import annotations

import torch
from torch import nn

from ._common import (
    TemporalAttentionScorer,
    TinyFrameEncoder,
    scores_to_mask,
)

_VARIANTS: dict[str, dict[str, int]] = {
    "memorytokensum_tiny": {"width": 24, "depth": 2},
    "memorytokensum_small": {"width": 32, "depth": 3},
    "memorytokensum_base": {"width": 48, "depth": 4},
}


class MemorytokensumVideoSummarizer(nn.Module):
    """Temporal summarizer with learned memory tokens and bidirectional memory reads."""

    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = TinyFrameEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        self.scorer = TemporalAttentionScorer(
            dim=int(self.encoder.out_dim),
            heads=4,
            depth=max(1, int(depth) - 1),
            dropout=float(dropout),
        )
        dim = int(self.encoder.out_dim)
        self.memory_tokens = nn.Parameter(torch.randn(max(2, int(depth)), dim) * 0.02)
        self.memory_read = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=4,
            dropout=float(dropout),
            batch_first=True,
        )
        self.frame_read = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=4,
            dropout=float(dropout),
            batch_first=True,
        )
        self.memory_norm = nn.LayerNorm(dim)
        self.frame_norm = nn.LayerNorm(dim)

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        memory = self.memory_tokens.unsqueeze(0).expand(int(feat.shape[0]), -1, -1)
        memory_update, _ = self.memory_read(memory, feat, feat, need_weights=False)
        memory = self.memory_norm(memory + memory_update)
        frame_context, _ = self.frame_read(feat, memory, memory, need_weights=False)
        conditioned = self.frame_norm(feat + frame_context)
        scores = torch.sigmoid(self.scorer(conditioned))
        return {"scores": scores, "summary_mask": scores_to_mask(scores)}


def build_memorytokensum_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "memorytokensum_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    cfg = _VARIANTS[str(variant).lower().strip()]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return MemorytokensumVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_memorytokensum_video_summarizer(
        in_channels=3, variant="memorytokensum_tiny", width_mult=0.5
    )
    out = m(x)
    print(
        "memorytokensum_tiny",
        {k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)},
    )
    loss = sum(v.mean() for v in out.values() if isinstance(v, torch.Tensor))
    loss.backward()
    print("ok")
