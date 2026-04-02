from __future__ import annotations

import torch
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "maam_tiny": {"width": 24, "depth": 2},
    "maam_small": {"width": 32, "depth": 3},
    "maam_base": {"width": 48, "depth": 4},
}


class MAAMVideoSummarizer(nn.Module):
    """Multi-annotation attention summarizer with latent annotator aggregation."""

    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_annotators = max(3, int(depth) + 1)
        self.encoder = TinyFrameEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        dim = int(self.encoder.out_dim)
        hidden = max(32, dim // 2)
        self.temporal = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=4,
            dropout=float(dropout),
            batch_first=True,
        )
        self.annotator_head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, self.num_annotators),
        )
        self.annotator_mixer = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.num_annotators),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        attn_feat, attn_map = self.temporal(feat, feat, feat, need_weights=True)
        annotator_logits = self.annotator_head(attn_feat)  # (B,T,A)
        annotator_scores = torch.sigmoid(annotator_logits).transpose(1, 2)  # (B,A,T)
        annotator_weights = torch.softmax(self.annotator_mixer(attn_feat.mean(dim=1)), dim=-1)  # (B,A)
        latent_logits = torch.einsum("ba,bat->bt", annotator_weights, annotator_scores)
        scores = latent_logits.clamp(0.0, 1.0)
        summary_mask = scores_to_mask(scores)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "annotator_scores": annotator_scores,
            "annotator_weights": annotator_weights,
            "attention_map": attn_map,
        }


def build_maam_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "maam_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MAAM variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return MAAMVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_maam_video_summarizer(in_channels=3, variant="maam_tiny", width_mult=0.5)
    out = m(x)
    print("maam_tiny", tuple(out["scores"].shape), tuple(out["annotator_scores"].shape))
    loss = out["scores"].mean() + out["annotator_weights"].mean()
    loss.backward()
    print("ok")
