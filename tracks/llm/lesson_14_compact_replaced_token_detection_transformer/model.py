from dataclasses import dataclass

import torch
from torch import nn


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(embed_dim // num_heads)
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        self.out = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.dropout = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scale = float(self.head_dim) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal = torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal.view(1, 1, seq_len, seq_len), -1e9)
        key_mask = attention_mask.to(torch.bool).view(bsz, 1, 1, seq_len)
        scores = scores.masked_fill(~key_mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        out = self.out(out)
        return out * attention_mask.to(torch.float32).unsqueeze(-1)


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(int(embed_dim))
        self.attn = MultiHeadCausalSelfAttention(
            embed_dim=int(embed_dim), num_heads=int(num_heads), dropout=float(dropout)
        )
        self.drop1 = nn.Dropout(p=float(dropout))
        self.ln2 = nn.LayerNorm(int(embed_dim))
        self.ff = nn.Sequential(
            nn.Linear(int(embed_dim), int(ff_dim)),
            nn.ReLU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(ff_dim), int(embed_dim)),
        )
        self.drop2 = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        x = x + self.drop1(self.attn(h, attention_mask=attention_mask))
        h = self.ln2(x)
        x = x + self.drop2(self.ff(h))
        return x


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    embed_dim: int = 128
    num_heads: int = 4
    ff_dim: int = 256
    dropout: float = 0.1


class CompactReplacedTokenDetectionTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(
            int(cfg.vocab_size), int(cfg.embed_dim), padding_idx=int(cfg.pad_id)
        )
        self.pos_embed = nn.Embedding(int(cfg.max_length), int(cfg.embed_dim))
        self.dropout = nn.Dropout(p=float(cfg.dropout))
        self.block = DecoderBlock(
            embed_dim=int(cfg.embed_dim),
            num_heads=int(cfg.num_heads),
            ff_dim=int(cfg.ff_dim),
            dropout=float(cfg.dropout),
        )
        self.ln = nn.LayerNorm(int(cfg.embed_dim))
        self.token_head = nn.Linear(int(cfg.embed_dim), int(cfg.vocab_size), bias=False)
        self.rtd_head = nn.Linear(int(cfg.embed_dim), 1, bias=True)

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = inputs["input_ids"].to(torch.long)
        attention_mask = inputs["attention_mask"].to(torch.float32)
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
        x = self.block(x, attention_mask=attention_mask)
        x = self.ln(x)
        token_logits = self.token_head(x)
        rtd_logits = self.rtd_head(x).squeeze(-1)
        return {"token_logits": token_logits, "rtd_logits": rtd_logits}


__all__ = ["ModelConfig", "CompactReplacedTokenDetectionTransformer"]
