from dataclasses import dataclass

import torch
from torch import nn


def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
    mask = torch.ones((int(length), int(length)), device=device, dtype=torch.bool)
    return torch.triu(mask, diagonal=1)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    max_src_len: int
    max_tgt_len: int
    embed_dim: int = 64
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1


class ToyTransformerSummarizer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(
            int(cfg.vocab_size), int(cfg.embed_dim), padding_idx=int(cfg.pad_id)
        )
        self.src_pos_embed = nn.Embedding(int(cfg.max_src_len), int(cfg.embed_dim))
        self.tgt_pos_embed = nn.Embedding(int(cfg.max_tgt_len), int(cfg.embed_dim))
        self.dropout = nn.Dropout(float(cfg.dropout))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(cfg.embed_dim),
            nhead=int(cfg.num_heads),
            dim_feedforward=int(cfg.ff_dim),
            dropout=float(cfg.dropout),
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=int(cfg.embed_dim),
            nhead=int(cfg.num_heads),
            dim_feedforward=int(cfg.ff_dim),
            dropout=float(cfg.dropout),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=int(cfg.num_encoder_layers)
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=int(cfg.num_decoder_layers)
        )
        self.ln = nn.LayerNorm(int(cfg.embed_dim))
        self.out = nn.Linear(int(cfg.embed_dim), int(cfg.vocab_size))

    def _embed_src(self, src_ids: torch.Tensor) -> torch.Tensor:
        b, s = src_ids.shape
        pos = torch.arange(s, device=src_ids.device).unsqueeze(0).expand(b, s)
        return self.dropout(self.token_embed(src_ids) + self.src_pos_embed(pos))

    def _embed_tgt(self, tgt_in_ids: torch.Tensor) -> torch.Tensor:
        b, t = tgt_in_ids.shape
        pos = torch.arange(t, device=tgt_in_ids.device).unsqueeze(0).expand(b, t)
        return self.dropout(self.token_embed(tgt_in_ids) + self.tgt_pos_embed(pos))

    def encode(self, *, src_ids: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src_key_padding_mask = ~src_mask.to(torch.bool)
        src = self._embed_src(src_ids.to(torch.long))
        return self.encoder(src, src_key_padding_mask=src_key_padding_mask)

    def forward(
        self, *, src_ids: torch.Tensor, src_mask: torch.Tensor, tgt_in_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        src_ids = src_ids.to(torch.long)
        tgt_in_ids = tgt_in_ids.to(torch.long)
        src_key_padding_mask = ~src_mask.to(torch.bool)
        tgt_key_padding_mask = tgt_in_ids.eq(int(self.cfg.pad_id))

        memory = self.encode(src_ids=src_ids, src_mask=src_mask)
        tgt = self._embed_tgt(tgt_in_ids)
        tgt_mask = _causal_mask(tgt_in_ids.shape[1], tgt_in_ids.device)
        hidden = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.out(self.ln(hidden))
        return {"logits": logits}

    @torch.no_grad()
    def greedy_decode(
        self, *, src_ids: torch.Tensor, src_mask: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        src_ids = src_ids.to(torch.long)
        memory = self.encode(src_ids=src_ids, src_mask=src_mask)
        src_key_padding_mask = ~src_mask.to(torch.bool)

        batch_size = int(src_ids.shape[0])
        generated = torch.full(
            (batch_size, int(max_len)),
            fill_value=int(self.cfg.pad_id),
            device=src_ids.device,
            dtype=torch.long,
        )
        generated[:, 0] = int(self.cfg.bos_id)

        for step in range(1, int(max_len)):
            cur = generated[:, :step]
            tgt = self._embed_tgt(cur)
            tgt_mask = _causal_mask(step, src_ids.device)
            hidden = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=cur.eq(int(self.cfg.pad_id)),
                memory_key_padding_mask=src_key_padding_mask,
            )
            next_token = self.out(self.ln(hidden[:, -1, :])).argmax(dim=-1)
            generated[:, step] = next_token

        return generated


__all__ = ["ModelConfig", "ToyTransformerSummarizer"]
