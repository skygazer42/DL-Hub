from __future__ import annotations

import math

import torch
from torch import nn

from ._common import (
    TemporalGRUScorer,
    TinyFrameEncoder,
    scores_to_mask,
)

_VARIANTS: dict[str, dict[str, int]] = {
    "queryfocus_sum_tiny": {"width": 24, "depth": 2},
    "queryfocus_sum_small": {"width": 32, "depth": 3},
    "queryfocus_sum_base": {"width": 48, "depth": 4},
}


class QueryfocusSumVideoSummarizer(nn.Module):
    """Query-conditioned frame scorer with a lightweight learned fallback prompt."""

    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = TinyFrameEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        self.scorer = TemporalGRUScorer(
            dim=int(self.encoder.out_dim),
            hidden_dim=max(16, int(self.encoder.out_dim)),
            layers=max(1, int(depth) - 1),
            dropout=float(dropout),
        )
        dim = int(self.encoder.out_dim)
        self.query_prompt = nn.Parameter(torch.randn(dim) * 0.02)
        self.query_proj = nn.Linear(dim, dim)
        self.frame_proj = nn.Linear(dim, dim)

    def forward(
        self,
        video: torch.Tensor,
        query: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        b, _, d = feat.shape
        if query is None:
            query_vec = self.query_prompt.unsqueeze(0).expand(int(b), -1)
        else:
            query_vec = query.to(device=feat.device, dtype=feat.dtype)
            if query_vec.ndim == 1:
                query_vec = query_vec.unsqueeze(0)
            elif query_vec.ndim == 3:
                query_vec = query_vec.mean(dim=1)
            elif query_vec.ndim != 2:
                raise ValueError(
                    f"query must have shape (D), (B,D) or (B,Q,D), got {tuple(query_vec.shape)}"
                )
            if int(query_vec.shape[0]) == 1 and int(b) > 1:
                query_vec = query_vec.expand(int(b), -1)
            elif int(query_vec.shape[0]) != int(b):
                raise ValueError(
                    f"query batch {int(query_vec.shape[0])} does not match video batch {int(b)}"
                )
            if int(query_vec.shape[-1]) < int(d):
                pad = torch.zeros(
                    int(b),
                    int(d) - int(query_vec.shape[-1]),
                    device=feat.device,
                    dtype=feat.dtype,
                )
                query_vec = torch.cat([query_vec, pad], dim=-1)
            elif int(query_vec.shape[-1]) > int(d):
                query_vec = query_vec[..., : int(d)]

        query_state = torch.tanh(self.query_proj(query_vec))
        alignment = torch.einsum("btd,bd->bt", self.frame_proj(feat), query_state) / math.sqrt(
            max(1, int(d))
        )
        conditioned = feat * (1.0 + torch.tanh(query_state).unsqueeze(1))
        scores = torch.sigmoid(self.scorer(conditioned) + alignment)
        return {"scores": scores, "summary_mask": scores_to_mask(scores)}


def build_queryfocus_sum_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "queryfocus_sum_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    cfg = _VARIANTS[str(variant).lower().strip()]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return QueryfocusSumVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_queryfocus_sum_video_summarizer(
        in_channels=3, variant="queryfocus_sum_tiny", width_mult=0.5
    )
    out = m(x, torch.randn(2, m.encoder.out_dim))
    print(
        "queryfocus_sum_tiny",
        {k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)},
    )
    loss = sum(v.mean() for v in out.values() if isinstance(v, torch.Tensor))
    loss.backward()
    print("ok")
