from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    embed_dim: int = 32
    hidden_dim: int = 48
    dropout: float = 0.1


class DenoisingSeq2Seq(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(
            num_embeddings=int(cfg.vocab_size),
            embedding_dim=int(cfg.embed_dim),
            padding_idx=int(cfg.pad_id),
        )
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.encoder = nn.GRU(
            input_size=int(cfg.embed_dim),
            hidden_size=int(cfg.hidden_dim),
            batch_first=True,
        )
        self.decoder = nn.GRU(
            input_size=int(cfg.embed_dim),
            hidden_size=int(cfg.hidden_dim),
            batch_first=True,
        )
        self.head = nn.Linear(int(cfg.hidden_dim), int(cfg.vocab_size))

    def encode(self, *, src_ids: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        lengths = src_mask.sum(dim=1).to(torch.long).clamp(min=1).cpu()
        emb = self.dropout(self.embedding(src_ids.to(torch.long)))
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _enc_out, hidden = self.encoder(packed)
        return hidden

    def forward(
        self, *, src_ids: torch.Tensor, src_mask: torch.Tensor, tgt_in_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode(src_ids=src_ids, src_mask=src_mask)
        tgt_emb = self.dropout(self.embedding(tgt_in_ids.to(torch.long)))
        dec_out, _ = self.decoder(tgt_emb, hidden)
        return {"logits": self.head(dec_out)}

    @torch.no_grad()
    def greedy_decode(self, *, src_ids: torch.Tensor, src_mask: torch.Tensor, max_len: int) -> torch.Tensor:
        hidden = self.encode(src_ids=src_ids, src_mask=src_mask)
        batch_size = int(src_ids.shape[0])
        current = torch.full(
            (batch_size, 1),
            fill_value=int(self.cfg.bos_id),
            device=src_ids.device,
            dtype=torch.long,
        )
        outputs: list[torch.Tensor] = []
        for _ in range(int(max_len)):
            step_out, hidden = self.decoder(self.embedding(current), hidden)
            logits = self.head(step_out[:, -1:, :])
            current = logits.argmax(dim=-1)
            outputs.append(current)
        return torch.cat(outputs, dim=1)


def reconstruction_token_accuracy(logits: torch.Tensor, labels: torch.Tensor, pad_id: int) -> float:
    with torch.no_grad():
        mask = labels.ne(int(pad_id))
        count = int(mask.sum().item())
        if count == 0:
            return 0.0
        preds = logits.argmax(dim=-1)
        correct = preds.eq(labels) & mask
        return float(correct.sum().item()) / float(count)


__all__ = ["DenoisingSeq2Seq", "ModelConfig", "reconstruction_token_accuracy"]
