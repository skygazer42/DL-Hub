from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ._common import TemporalGRUScorer, TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "lgrln_tiny": {"width": 24, "depth": 2},
    "lgrln_small": {"width": 32, "depth": 3},
    "lgrln_base": {"width": 48, "depth": 4},
}


def _prepare_language_state(
    query: torch.Tensor | None,
    *,
    batch: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    prompt: torch.Tensor,
) -> torch.Tensor:
    if query is None:
        return prompt.mean(dim=0, keepdim=True).expand(int(batch), -1)

    q = query.to(device=device, dtype=dtype)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    elif q.ndim == 3:
        q = q.mean(dim=1)
    elif q.ndim != 2:
        raise ValueError(f"query must have shape (D), (B,D) or (B,Q,D), got {tuple(q.shape)}")

    if int(q.shape[0]) == 1 and int(batch) > 1:
        q = q.expand(int(batch), -1)
    elif int(q.shape[0]) != int(batch):
        raise ValueError(f"query batch {int(q.shape[0])} does not match video batch {int(batch)}")

    cur_dim = int(q.shape[-1])
    if cur_dim < int(dim):
        pad = torch.zeros(int(batch), int(dim) - cur_dim, device=device, dtype=dtype)
        q = torch.cat([q, pad], dim=-1)
    elif cur_dim > int(dim):
        q = q[..., : int(dim)]
    return q


class LGRLNSummarizer(nn.Module):
    """Language-guided relation learning summarizer with fallback prompts."""

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
        num_prompt = max(2, int(depth))

        self.language_prompt = nn.Parameter(torch.randn(num_prompt, dim) * 0.02)
        self.language_proj = nn.Linear(dim, dim)
        self.relation_proj = nn.Linear(dim, dim)
        self.scorer = TemporalGRUScorer(
            dim=dim * 3,
            hidden_dim=hidden,
            layers=1,
            dropout=float(dropout),
        )

    def forward(
        self,
        video: torch.Tensor,
        query: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        b, t, d = feat.shape
        lang = _prepare_language_state(
            query,
            batch=int(b),
            dim=int(d),
            device=feat.device,
            dtype=feat.dtype,
            prompt=self.language_prompt.to(device=feat.device, dtype=feat.dtype),
        )
        lang = self.language_proj(lang)
        lang_expand = lang.unsqueeze(1).expand(-1, int(t), -1)

        guided = feat * torch.sigmoid(lang_expand)
        rel_feat = self.relation_proj(feat + 0.1 * lang_expand)
        relation_scores = torch.softmax(
            torch.matmul(rel_feat, rel_feat.transpose(1, 2)) / math.sqrt(max(1, int(d))),
            dim=-1,
        )
        relation_ctx = torch.matmul(relation_scores, feat)
        fused = torch.cat([feat, relation_ctx, guided], dim=-1)
        raw_scores = self.scorer(fused)
        language_alignment = F.normalize(feat, dim=-1).mul(
            F.normalize(lang_expand, dim=-1)
        ).sum(dim=-1)
        scores = torch.sigmoid(raw_scores + language_alignment)
        summary_mask = scores_to_mask(scores)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "relation_scores": relation_scores,
            "language_alignment": language_alignment,
        }


def build_lgrln_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "lgrln_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown LGRLN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return LGRLNSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_lgrln_video_summarizer(
        in_channels=3,
        variant="lgrln_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("lgrln_tiny", tuple(out["scores"].shape), tuple(out["relation_scores"].shape))
    loss = out["scores"].mean() + out["relation_scores"].mean()
    loss.backward()
    print("ok")
