from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    embed_dim: int = 64
    hidden_dim: int = 128
    num_tags: int = 7
    dropout: float = 0.1


class BiLstmNerTagger(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(
            int(cfg.vocab_size), int(cfg.embed_dim), padding_idx=int(cfg.pad_id)
        )
        self.dropout = nn.Dropout(p=float(cfg.dropout))
        self.lstm = nn.LSTM(
            input_size=int(cfg.embed_dim),
            hidden_size=int(cfg.hidden_dim),
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Linear(int(cfg.hidden_dim) * 2, int(cfg.num_tags))

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"]  # (B, T)
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids shape (B, T), got {tuple(input_ids.shape)}")

        x = self.embed(input_ids)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        logits = self.head(x)  # (B, T, num_tags)
        return logits


__all__ = ["BiLstmNerTagger", "ModelConfig"]
