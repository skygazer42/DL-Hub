from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_docs: int
    embed_dim: int = 128
    hidden_dim: int = 192
    dropout: float = 0.1


class ToyRagLanguageModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(
            int(cfg.vocab_size),
            int(cfg.embed_dim),
            padding_idx=int(cfg.pad_id),
        )
        self.pos_embed = nn.Embedding(int(cfg.max_length), int(cfg.embed_dim))
        self.doc_embed = nn.Embedding(int(cfg.num_docs), int(cfg.embed_dim))
        self.dropout = nn.Dropout(p=float(cfg.dropout))

        self.doc_to_hidden = nn.Linear(int(cfg.embed_dim), int(cfg.hidden_dim))
        self.decoder = nn.GRU(
            input_size=int(cfg.embed_dim),
            hidden_size=int(cfg.hidden_dim),
            batch_first=True,
        )
        self.head = nn.Linear(int(cfg.hidden_dim), int(cfg.vocab_size), bias=False)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        doc_ids: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.to(torch.long)
        attention_mask = attention_mask.to(torch.float32)
        doc_ids = doc_ids.to(torch.long)

        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids shape (B, T), got {tuple(input_ids.shape)}")
        if doc_ids.ndim != 1:
            raise ValueError(f"Expected doc_ids shape (B,), got {tuple(doc_ids.shape)}")

        b, t = input_ids.shape
        if t != int(self.cfg.max_length):
            raise ValueError(
                f"Expected max_length={int(self.cfg.max_length)} tokens, got sequence length {int(t)}"
            )
        if int(doc_ids.shape[0]) != int(b):
            raise ValueError(
                f"Expected doc_ids batch dimension {int(b)}, got {int(doc_ids.shape[0])}"
            )

        pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        token_x = self.token_embed(input_ids)
        pos_x = self.pos_embed(pos)
        doc_x = self.doc_embed(doc_ids).unsqueeze(1).expand(b, t, -1)

        x = self.dropout(token_x + pos_x + doc_x)
        h0 = torch.tanh(self.doc_to_hidden(self.doc_embed(doc_ids))).unsqueeze(0)
        out, _ = self.decoder(x, h0)
        logits = self.head(out)
        return logits * attention_mask.unsqueeze(-1)


__all__ = ["ModelConfig", "ToyRagLanguageModel"]
