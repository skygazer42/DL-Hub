from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ._common import TemporalAttentionScorer, TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "llm_pretrain_tiny": {"width": 24, "depth": 2},
    "llm_pretrain_small": {"width": 32, "depth": 3},
    "llm_pretrain_base": {"width": 48, "depth": 4},
}


class LLMPretrainVideoSummarizer(nn.Module):
    """LLM-oracle-inspired summarizer with prompt-token distillation."""

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
        num_tokens = max(6, int(depth) * 2)

        self.prompt_tokens = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.token_proj = nn.Linear(dim, dim)
        self.frame_proj = nn.Linear(dim, dim)
        self.sequence_scorer = TemporalAttentionScorer(
            dim=dim * 2,
            heads=4,
            depth=max(1, int(depth) - 1),
            dropout=float(dropout),
        )
        self.oracle_head = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        b, t, d = feat.shape
        tokens = self.prompt_tokens.to(device=feat.device, dtype=feat.dtype)
        frame_key = self.frame_proj(feat)
        token_key = self.token_proj(tokens).unsqueeze(0).expand(int(b), -1, -1)

        oracle_attn = torch.softmax(
            torch.einsum("btd,bkd->btk", frame_key, token_key) / math.sqrt(max(1, int(d))),
            dim=-1,
        )
        oracle_context = torch.einsum("btk,bkd->btd", oracle_attn, tokens.unsqueeze(0).expand(int(b), -1, -1))
        sequence_feat = torch.cat([feat, oracle_context], dim=-1)

        sequence_logits = self.sequence_scorer(sequence_feat)
        oracle_logits = self.oracle_head(sequence_feat).squeeze(-1)
        oracle_alignment = (F.normalize(frame_key, dim=-1) * F.normalize(oracle_context, dim=-1)).sum(dim=-1)
        scores = torch.sigmoid(sequence_logits + 0.35 * oracle_logits + 0.20 * oracle_alignment)
        summary_mask = scores_to_mask(scores)
        summary_prior = oracle_attn.mean(dim=1)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "oracle_attn": oracle_attn,
            "summary_prior": summary_prior,
        }


def build_llm_pretrain_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "llm_pretrain_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown LLM-Pretrain variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return LLMPretrainVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_llm_pretrain_video_summarizer(
        in_channels=3,
        variant="llm_pretrain_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("llm_pretrain_tiny", tuple(out["scores"].shape), tuple(out["oracle_attn"].shape))
    loss = out["scores"].mean() + out["summary_prior"].mean()
    loss.backward()
    print("ok")
