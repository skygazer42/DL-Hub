
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
        b, t, c = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        scale = float(self.head_dim) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)

        # Causal mask: allow attending to <= current position.
        causal = torch.ones((t, t), device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal.view(1, 1, t, t), -1e9)

        # Mask padded keys.
        key_mask = attention_mask.to(torch.bool).view(b, 1, 1, t)
        scores = scores.masked_fill(~key_mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        out = self.out(out)

        # Zero padded queries.
        out = out * attention_mask.to(torch.float32).unsqueeze(-1)
        return out


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(int(embed_dim))
        self.attn = MultiHeadCausalSelfAttention(embed_dim=int(embed_dim), num_heads=int(num_heads), dropout=dropout)
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
    num_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1


class CausalTransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(int(cfg.vocab_size), int(cfg.embed_dim), padding_idx=int(cfg.pad_id))
        self.pos_embed = nn.Embedding(int(cfg.max_length), int(cfg.embed_dim))
        self.dropout = nn.Dropout(p=float(cfg.dropout))

        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    embed_dim=int(cfg.embed_dim),
                    num_heads=int(cfg.num_heads),
                    ff_dim=int(cfg.ff_dim),
                    dropout=float(cfg.dropout),
                )
                for _ in range(int(cfg.num_layers))
            ]
        )
        self.ln = nn.LayerNorm(int(cfg.embed_dim))
        self.head = nn.Linear(int(cfg.embed_dim), int(cfg.vocab_size), bias=False)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"].to(torch.long)  # (B, T)
        attention_mask = inputs["attention_mask"].to(torch.float32)  # (B, T) in {0,1}

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

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.ln(x)
        logits = self.head(x)  # (B, T, V)
        return logits


__all__ = ["CausalTransformerLM", "ModelConfig"]

