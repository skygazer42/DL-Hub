from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import TemporalGRUScorer, TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "contrast_sum_tiny": {"width": 24, "depth": 2},
    "contrast_sum_small": {"width": 32, "depth": 3},
    "contrast_sum_base": {"width": 48, "depth": 4},
}


class ContrastSumVideoSummarizer(nn.Module):
    """Contrastive summarizer with paired temporal views."""

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
        self.anchor_scorer = TemporalGRUScorer(
            dim=dim, hidden_dim=hidden, layers=1, dropout=float(dropout)
        )
        self.pair_head = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        b, t, _ = feat.shape

        view2 = 0.85 * feat + 0.15 * torch.roll(feat, shifts=1, dims=1)
        norm_feat = F.normalize(feat, dim=-1)
        norm_view2 = F.normalize(view2, dim=-1)
        contrastive_affinity = torch.matmul(norm_feat, norm_view2.transpose(1, 2))
        positive_alignment = contrastive_affinity.diagonal(dim1=1, dim2=2)

        novelty = torch.zeros(int(b), int(t), device=feat.device, dtype=feat.dtype)
        if int(t) > 1:
            novelty[:, 1:] = (norm_feat[:, 1:] - norm_feat[:, :-1]).pow(2).mean(dim=-1).sqrt()

        anchor_logits = self.anchor_scorer(feat)
        fused = torch.cat([feat, view2, (feat - view2).abs()], dim=-1)
        pair_logits = self.pair_head(fused).squeeze(-1)
        scores = torch.sigmoid(
            pair_logits + 0.30 * anchor_logits + 0.25 * positive_alignment + 0.15 * novelty
        )
        summary_mask = scores_to_mask(scores)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "contrastive_affinity": contrastive_affinity,
            "positive_alignment": positive_alignment,
        }


def build_contrast_sum_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "contrast_sum_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown Contrast-SUM variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return ContrastSumVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_contrast_sum_video_summarizer(
        in_channels=3,
        variant="contrast_sum_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("contrast_sum_tiny", tuple(out["scores"].shape), tuple(out["contrastive_affinity"].shape))
    loss = out["scores"].mean() + out["positive_alignment"].mean()
    loss.backward()
    print("ok")
