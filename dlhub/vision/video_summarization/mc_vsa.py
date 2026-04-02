from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "mc_vsa_tiny": {"width": 24, "depth": 2},
    "mc_vsa_small": {"width": 32, "depth": 3},
    "mc_vsa_base": {"width": 48, "depth": 4},
}


class MCVSAVideoSummarizer(nn.Module):
    """Multi-concept video summarizer with concept-attended scoring."""

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
        self.num_concepts = max(4, int(depth) + 2)
        self.concept_bank = nn.Parameter(torch.randn(self.num_concepts, dim) * 0.02)
        self.head = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        b, t, _ = feat.shape
        bank = self.concept_bank.to(device=feat.device, dtype=feat.dtype)
        norm_feat = F.normalize(feat, dim=-1)
        norm_bank = F.normalize(bank, dim=-1)

        concept_attn = torch.softmax(torch.matmul(norm_feat, norm_bank.transpose(0, 1)), dim=-1)  # (B,T,K)
        concept_context = torch.matmul(concept_attn, bank)
        concept_scores = torch.sigmoid(
            torch.einsum("btd,kd->btk", norm_feat, norm_bank)
        ).transpose(1, 2)  # (B,K,T)

        concept_weights = torch.softmax(concept_attn.mean(dim=1), dim=-1)
        weighted_concept_score = torch.einsum("bk,bkt->bt", concept_weights, concept_scores)
        raw_scores = self.head(torch.cat([feat, concept_context], dim=-1)).squeeze(-1)
        scores = torch.sigmoid(raw_scores + 0.35 * weighted_concept_score)
        summary_mask = scores_to_mask(scores)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "concept_attn": concept_attn,
            "concept_scores": concept_scores,
        }


def build_mc_vsa_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "mc_vsa_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MC-VSA variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return MCVSAVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_mc_vsa_video_summarizer(in_channels=3, variant="mc_vsa_tiny", width_mult=0.5)
    out = m(x)
    print("mc_vsa_tiny", tuple(out["scores"].shape), tuple(out["concept_scores"].shape))
    loss = out["scores"].mean() + out["concept_attn"].mean()
    loss.backward()
    print("ok")
