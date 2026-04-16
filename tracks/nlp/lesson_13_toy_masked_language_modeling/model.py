from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from tracks.nlp.lesson_02_toy_text_classification_transformer.model import TransformerEncoderBlock


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1


def masked_token_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, *, ignore_index: int = -100
) -> float:
    with torch.no_grad():
        mask = labels.ne(int(ignore_index))
        count = int(mask.sum().item())
        if count == 0:
            return 0.0
        preds = logits.argmax(dim=-1)
        correct = preds.eq(labels) & mask
        return float(correct.sum().item()) / float(count)


class ToyMaskedLanguageModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(
            int(cfg.vocab_size), int(cfg.embed_dim), padding_idx=int(cfg.pad_id)
        )
        self.pos_embed = nn.Embedding(int(cfg.max_length), int(cfg.embed_dim))
        self.dropout = nn.Dropout(p=float(cfg.dropout))
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=int(cfg.embed_dim),
                    num_heads=int(cfg.num_heads),
                    ff_dim=int(cfg.ff_dim),
                    dropout=float(cfg.dropout),
                )
                for _ in range(int(cfg.num_layers))
            ]
        )
        self.ln = nn.LayerNorm(int(cfg.embed_dim))
        self.head = nn.Linear(int(cfg.embed_dim), int(cfg.vocab_size))

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids shape (B, T), got {tuple(input_ids.shape)}")

        bsz, seq_len = input_ids.shape
        if seq_len != int(self.cfg.max_length):
            raise ValueError(
                f"Expected max_length={int(self.cfg.max_length)} tokens, got sequence length {int(seq_len)}"
            )

        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.ln(x)
        return {"logits": self.head(x)}


__all__ = ["ModelConfig", "ToyMaskedLanguageModel", "masked_token_accuracy"]
