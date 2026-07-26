from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import TemporalGRUScorer, TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "progressive_ssl_tiny": {"width": 24, "depth": 2},
    "progressive_ssl_small": {"width": 32, "depth": 3},
    "progressive_ssl_base": {"width": 48, "depth": 4},
}


class ProgressiveSSLVideoSummarizer(nn.Module):
    """Progressive self-supervised summarizer with latent concept prompts."""

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
        num_concepts = max(4, int(depth) + 2)
        self.concept_bank = nn.Parameter(torch.randn(num_concepts, dim) * 0.02)
        self.stage1 = TemporalGRUScorer(
            dim=dim, hidden_dim=hidden, layers=1, dropout=float(dropout)
        )
        self.refine = nn.Sequential(
            nn.Linear(dim * 2 + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        b, t, d = feat.shape
        bank = self.concept_bank.to(device=feat.device, dtype=feat.dtype)
        norm_feat = F.normalize(feat, dim=-1)
        norm_bank = F.normalize(bank, dim=-1)

        concept_attn = torch.softmax(torch.matmul(norm_feat, norm_bank.transpose(0, 1)), dim=-1)
        concept_context = torch.matmul(concept_attn, bank)

        stage1_logits = self.stage1(feat)
        stage1_scores = torch.sigmoid(stage1_logits)

        temporal_teacher = torch.zeros(int(b), int(t), 1, device=feat.device, dtype=feat.dtype)
        temporal_teacher[:, :, 0] = stage1_scores
        if int(t) > 1:
            temporal_teacher[:, 1:, 0] = (
                0.5 * temporal_teacher[:, 1:, 0] + 0.5 * stage1_scores[:, :-1]
            )

        refined_feat = torch.cat([feat, concept_context, temporal_teacher], dim=-1)
        refined_logits = self.refine(refined_feat).squeeze(-1)
        concept_alignment = (norm_feat * F.normalize(concept_context, dim=-1)).sum(dim=-1)
        scores = torch.sigmoid(refined_logits + 0.25 * stage1_logits + 0.20 * concept_alignment)
        summary_mask = scores_to_mask(scores)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "stage1_scores": stage1_scores,
            "concept_attn": concept_attn,
        }


def build_progressive_ssl_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "progressive_ssl_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown Progressive-SSL variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return ProgressiveSSLVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_progressive_ssl_video_summarizer(
        in_channels=3,
        variant="progressive_ssl_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("progressive_ssl_tiny", tuple(out["scores"].shape), tuple(out["concept_attn"].shape))
    loss = out["scores"].mean() + out["stage1_scores"].mean()
    loss.backward()
    print("ok")
