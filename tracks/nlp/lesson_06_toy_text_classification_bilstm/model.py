
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    embed_dim: int = 64
    hidden_dim: int = 64
    num_classes: int = 2
    dropout: float = 0.2


class BiLSTMTextClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(
            num_embeddings=int(cfg.vocab_size),
            embedding_dim=int(cfg.embed_dim),
            padding_idx=int(cfg.pad_id),
        )
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.encoder = nn.LSTM(
            input_size=int(cfg.embed_dim),
            hidden_size=int(cfg.hidden_dim),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(2 * int(cfg.hidden_dim), int(cfg.num_classes))

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"].to(torch.long)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != int(self.cfg.pad_id)).to(torch.float32)
        lengths = attention_mask.sum(dim=1).to(torch.long).clamp(min=1).cpu()

        emb = self.dropout(self.embedding(input_ids))
        packed = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.encoder(packed)  # h_n: (2, B, H) for 1-layer BiLSTM
        h = torch.cat([h_n[0], h_n[1]], dim=1)
        h = self.dropout(h)
        return self.classifier(h)


__all__ = ["BiLSTMTextClassifier", "ModelConfig"]

