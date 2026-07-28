from dataclasses import dataclass

import torch
from torch import nn


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(dtype=torch.float32)
    logits = logits.masked_fill(mask <= 0, -1e9)
    return torch.softmax(logits, dim=dim)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.w_enc = nn.Linear(int(hidden_dim), int(hidden_dim), bias=False)
        self.w_dec = nn.Linear(int(hidden_dim), int(hidden_dim), bias=False)
        self.v = nn.Linear(int(hidden_dim), 1, bias=False)

    def forward(
        self, *, enc_out: torch.Tensor, dec_h: torch.Tensor, enc_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute context and weights.

        enc_out: (B, S, H)
        dec_h: (B, H)
        enc_mask: (B, S) float {0,1}
        """

        b, s, h = enc_out.shape
        dec = self.w_dec(dec_h).view(b, 1, h).expand(b, s, h)
        score = self.v(torch.tanh(self.w_enc(enc_out) + dec)).squeeze(-1)  # (B, S)
        attn = _masked_softmax(score, enc_mask, dim=1)  # (B, S)
        ctx = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)  # (B, H)
        return ctx, attn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int
    embed_dim: int = 64
    hidden_dim: int = 128
    dropout: float = 0.1


class Seq2SeqWithAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(
            int(cfg.vocab_size), int(cfg.embed_dim), padding_idx=int(cfg.pad_id)
        )
        self.emb_drop = nn.Dropout(float(cfg.dropout))

        self.encoder = nn.GRU(
            input_size=int(cfg.embed_dim),
            hidden_size=int(cfg.hidden_dim),
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.attn = BahdanauAttention(int(cfg.hidden_dim))
        self.decoder_cell = nn.GRUCell(
            int(cfg.embed_dim) + int(cfg.hidden_dim), int(cfg.hidden_dim)
        )

        self.out = nn.Linear(int(cfg.hidden_dim) + int(cfg.hidden_dim), int(cfg.vocab_size))
        self.out_drop = nn.Dropout(float(cfg.dropout))

    def encode(
        self, *, src_ids: torch.Tensor, src_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_ids = src_ids.to(torch.long)
        src_mask = src_mask.to(torch.float32)
        lengths = src_mask.sum(dim=1).to(torch.long).clamp(min=1).cpu()

        emb = self.emb_drop(self.embed(src_ids))  # (B, S, E)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        enc_packed, h = self.encoder(packed)
        enc_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            enc_packed, batch_first=True, total_length=int(src_ids.shape[1])
        )
        # h: (1, B, H)
        return enc_out, h.squeeze(0)

    def forward(
        self, *, src_ids: torch.Tensor, src_mask: torch.Tensor, tgt_in_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        enc_out, dec_h = self.encode(src_ids=src_ids, src_mask=src_mask)
        enc_mask = src_mask.to(torch.float32)

        tgt_in_ids = tgt_in_ids.to(torch.long)
        b, t = tgt_in_ids.shape

        logits = torch.empty(
            (b, t, int(self.cfg.vocab_size)), device=tgt_in_ids.device, dtype=torch.float32
        )
        attn_weights = torch.empty(
            (b, t, int(enc_out.shape[1])), device=tgt_in_ids.device, dtype=torch.float32
        )

        for step in range(int(t)):
            token = tgt_in_ids[:, step]
            emb = self.emb_drop(self.embed(token))  # (B, E)
            ctx, attn = self.attn(enc_out=enc_out, dec_h=dec_h, enc_mask=enc_mask)
            dec_h = self.decoder_cell(torch.cat([emb, ctx], dim=1), dec_h)
            dec_h = self.out_drop(dec_h)
            out_step = self.out(torch.cat([dec_h, ctx], dim=1))
            logits[:, step, :] = out_step
            attn_weights[:, step, :] = attn

        return {"logits": logits, "attn": attn_weights}

    @torch.no_grad()
    def greedy_decode(
        self, *, src_ids: torch.Tensor, src_mask: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        enc_out, dec_h = self.encode(src_ids=src_ids, src_mask=src_mask)
        enc_mask = src_mask.to(torch.float32)

        b = int(src_ids.shape[0])
        out_ids = torch.full(
            (b, int(max_len)),
            fill_value=int(self.cfg.pad_id),
            device=src_ids.device,
            dtype=torch.long,
        )

        cur = torch.full(
            (b,), fill_value=int(self.cfg.bos_id), device=src_ids.device, dtype=torch.long
        )
        for t in range(int(max_len)):
            emb = self.embed(cur)
            ctx, _ = self.attn(enc_out=enc_out, dec_h=dec_h, enc_mask=enc_mask)
            dec_h = self.decoder_cell(torch.cat([emb, ctx], dim=1), dec_h)
            logits = self.out(torch.cat([dec_h, ctx], dim=1))
            nxt = logits.argmax(dim=1)
            out_ids[:, t] = nxt
            cur = nxt
        return out_ids


__all__ = ["Seq2SeqWithAttention", "ModelConfig"]
