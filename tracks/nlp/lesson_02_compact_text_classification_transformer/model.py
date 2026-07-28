from dataclasses import dataclass

import torch
from torch import nn


def _masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool `(B, T, C)` into `(B, C)` using mask `(B, T)` in {0,1}."""

    if mask.dtype != torch.float32:
        mask = mask.to(torch.float32)
    w = mask.unsqueeze(-1)  # (B, T, 1)
    summed = (x * w).sum(dim=1)  # (B, C)
    denom = w.sum(dim=1).clamp(min=1.0)
    return summed / denom


class MultiHeadSelfAttention(nn.Module):
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
        # x: (B, T, C)
        b, t, c = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, heads, T, head_dim)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        scale = float(self.head_dim) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, heads, T, T)

        # Mask padded keys.
        key_mask = attention_mask.to(torch.bool).view(b, 1, 1, t)
        scores = scores.masked_fill(~key_mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        out = self.out(out)

        # Zero out padded queries (optional but makes behavior explicit).
        query_mask = attention_mask.to(torch.float32).unsqueeze(-1)
        out = out * query_mask
        return out


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.drop1 = nn.Dropout(p=float(dropout))

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(ff_dim, embed_dim),
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
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1
    num_classes: int = 2


class TransformerTextClassifier(nn.Module):
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
        self.head = nn.Linear(int(cfg.embed_dim), int(cfg.num_classes))

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"]  # (B, T)
        attention_mask = inputs["attention_mask"]  # (B, T) float32 in {0,1}
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

        pooled = _masked_mean_pool(x, attention_mask)
        logits = self.head(pooled)
        return logits
