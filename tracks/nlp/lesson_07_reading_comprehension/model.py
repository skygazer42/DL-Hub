
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    embed_dim: int = 64
    hidden_dim: int = 64
    dropout: float = 0.1


class SimpleSpanQA(nn.Module):
    """A minimal span predictor: encode context/question, then score each context position."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(int(cfg.vocab_size), int(cfg.embed_dim), padding_idx=int(cfg.pad_id))
        self.embed_drop = nn.Dropout(float(cfg.dropout))

        self.ctx_enc = nn.LSTM(
            input_size=int(cfg.embed_dim),
            hidden_size=int(cfg.hidden_dim),
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.q_enc = nn.LSTM(
            input_size=int(cfg.embed_dim),
            hidden_size=int(cfg.hidden_dim),
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        enc_dim = 2 * int(cfg.hidden_dim)
        self.ctx_proj_start = nn.Linear(enc_dim, enc_dim)
        self.q_proj_start = nn.Linear(enc_dim, enc_dim, bias=False)
        self.ctx_proj_end = nn.Linear(enc_dim, enc_dim)
        self.q_proj_end = nn.Linear(enc_dim, enc_dim, bias=False)

        self.out_drop = nn.Dropout(float(cfg.dropout))

    def forward(
        self,
        *,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context_ids = context_ids.to(torch.long)
        question_ids = question_ids.to(torch.long)
        context_mask = context_mask.to(torch.float32)
        question_mask = question_mask.to(torch.float32)

        ctx = self.embed_drop(self.embed(context_ids))  # (B, Lc, E)
        q = self.embed_drop(self.embed(question_ids))  # (B, Lq, E)

        ctx_h, _ = self.ctx_enc(ctx)  # (B, Lc, 2H)
        q_h, _ = self.q_enc(q)  # (B, Lq, 2H)

        # Mean pooling over question tokens (mask-aware).
        q_w = question_mask.unsqueeze(-1)  # (B, Lq, 1)
        q_sum = (q_h * q_w).sum(dim=1)  # (B, 2H)
        q_den = q_w.sum(dim=1).clamp(min=1.0)  # (B, 1)
        q_vec = q_sum / q_den  # (B, 2H)
        q_vec = self.out_drop(q_vec)

        # Score each context position with a simple multiplicative interaction.
        ctx_s = self.ctx_proj_start(ctx_h)  # (B, Lc, 2H)
        q_s = self.q_proj_start(q_vec).unsqueeze(1)  # (B, 1, 2H)
        start_logits = (ctx_s * q_s).sum(dim=-1)  # (B, Lc)

        ctx_e = self.ctx_proj_end(ctx_h)
        q_e = self.q_proj_end(q_vec).unsqueeze(1)
        end_logits = (ctx_e * q_e).sum(dim=-1)

        # Mask padding positions.
        start_logits = start_logits.masked_fill(context_mask <= 0, -1e9)
        end_logits = end_logits.masked_fill(context_mask <= 0, -1e9)
        return {"start_logits": start_logits, "end_logits": end_logits}


__all__ = ["SimpleSpanQA", "ModelConfig"]

