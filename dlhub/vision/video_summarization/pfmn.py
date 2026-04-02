from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "pfmn_tiny": {"width": 24, "depth": 2},
    "pfmn_small": {"width": 32, "depth": 3},
    "pfmn_base": {"width": 48, "depth": 4},
}


class PFMNVideoSummarizer(nn.Module):
    """Past-future memory network summarizer."""

    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = TinyFrameEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        dim = int(self.encoder.out_dim)
        hidden = max(32, dim // 2)
        self.query_proj = nn.Linear(dim, dim)
        self.memory_proj = nn.Linear(dim, dim)
        self.head = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        _, t, d = feat.shape
        query = self.query_proj(feat)
        memory = self.memory_proj(feat)
        sim = torch.matmul(query, memory.transpose(1, 2)) / math.sqrt(max(1, int(d)))

        idx = torch.arange(int(t), device=feat.device)
        past_mask = idx.view(1, int(t), 1) > idx.view(1, 1, int(t))
        future_mask = idx.view(1, int(t), 1) < idx.view(1, 1, int(t))
        past_attn = torch.softmax(sim.masked_fill(~past_mask, -1e4), dim=-1)
        future_attn = torch.softmax(sim.masked_fill(~future_mask, -1e4), dim=-1)

        past_ctx = torch.matmul(past_attn, feat)
        future_ctx = torch.matmul(future_attn, feat)
        fallback = feat.mean(dim=1, keepdim=True).expand_as(feat)
        no_past = (idx == 0).view(1, int(t), 1)
        no_future = (idx == int(t) - 1).view(1, int(t), 1)
        past_ctx = torch.where(no_past, fallback, past_ctx)
        future_ctx = torch.where(no_future, fallback, future_ctx)

        fused = torch.cat([feat, past_ctx, future_ctx], dim=-1)
        memory_balance = F.cosine_similarity(past_ctx, future_ctx, dim=-1)
        raw_scores = self.head(fused).squeeze(-1)
        scores = torch.sigmoid(raw_scores + 0.20 * memory_balance)
        summary_mask = scores_to_mask(scores)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "past_attention": past_attn,
            "future_attention": future_attn,
        }


def build_pfmn_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "pfmn_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown PFMN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return PFMNVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_pfmn_video_summarizer(in_channels=3, variant="pfmn_tiny", width_mult=0.5)
    out = m(x)
    print("pfmn_tiny", tuple(out["scores"].shape), tuple(out["past_attention"].shape))
    loss = out["scores"].mean() + out["future_attention"].mean()
    loss.backward()
    print("ok")
