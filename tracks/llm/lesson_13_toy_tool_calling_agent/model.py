from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_tools: int
    embed_dim: int = 128
    hidden_dim: int = 192
    dropout: float = 0.1


class ToyToolCallingAgent(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(
            int(cfg.vocab_size),
            int(cfg.embed_dim),
            padding_idx=int(cfg.pad_id),
        )
        self.pos_embed = nn.Embedding(int(cfg.max_length), int(cfg.embed_dim))
        self.dropout = nn.Dropout(p=float(cfg.dropout))
        self.encoder = nn.GRU(
            input_size=int(cfg.embed_dim),
            hidden_size=int(cfg.hidden_dim),
            batch_first=True,
        )
        self.token_head = nn.Linear(int(cfg.hidden_dim), int(cfg.vocab_size), bias=False)
        self.tool_head = nn.Linear(int(cfg.hidden_dim), int(cfg.num_tools))

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        input_ids = input_ids.to(torch.long)
        attention_mask = attention_mask.to(torch.float32)

        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids shape (B, T), got {tuple(input_ids.shape)}")
        b, t = input_ids.shape
        if t != int(self.cfg.max_length):
            raise ValueError(
                f"Expected max_length={int(self.cfg.max_length)} tokens, got sequence length {int(t)}"
            )

        pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        x = self.dropout(x)
        hidden, _ = self.encoder(x)

        token_logits = self.token_head(hidden) * attention_mask.unsqueeze(-1)
        masked_hidden = hidden * attention_mask.unsqueeze(-1)
        denom = attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = masked_hidden.sum(dim=1) / denom
        tool_logits = self.tool_head(pooled)
        return {"token_logits": token_logits, "tool_logits": tool_logits}


__all__ = ["ModelConfig", "ToyToolCallingAgent"]
