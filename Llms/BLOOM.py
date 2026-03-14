from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ._shared import (
    build_alibi_bias,
    causal_mask,
    causal_mask_with_offset,
    expand_key_padding_mask,
    GELUMLP,
    make_attention_mask,
)


@dataclass(frozen=True)
class BloomConfig:
    vocab_size: int
    max_seq_len: int
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 6
    intermediate_size: int | None = None
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5


class BloomAttention(nn.Module):
    use_alibi = True

    def __init__(self, config: BloomConfig) -> None:
        super().__init__()
        if int(config.hidden_size) % int(config.num_heads) != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_heads)
        self.head_dim = int(config.hidden_size // config.num_heads)
        self.query_key_value = nn.Linear(self.hidden_size, self.hidden_size * 3)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(float(config.dropout))

    def forward_with_cache(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        bsz, seq_len, _ = x.shape
        past_len = 0
        qkv = self.query_key_value(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        query_mask = attention_mask
        if past_key_value is not None:
            past_k, past_v, past_mask = past_key_value
            past_len = int(past_k.shape[-2])
            k = torch.cat((past_k.to(device=k.device, dtype=k.dtype), k), dim=-2)
            v = torch.cat((past_v.to(device=v.device, dtype=v.dtype), v), dim=-2)
            attention_mask = torch.cat(
                (
                    past_mask.to(device=attention_mask.device, dtype=attention_mask.dtype),
                    attention_mask,
                ),
                dim=1,
            )
        key_len = int(k.shape[-2])

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        scores = scores + build_alibi_bias(
            self.num_heads,
            seq_len,
            device=x.device,
            key_len=key_len,
            query_offset=past_len,
        )
        scores = scores.masked_fill(
            ~expand_key_padding_mask(attention_mask, batch_size=bsz, seq_len=key_len),
            torch.finfo(scores.dtype).min,
        )
        causal = (
            causal_mask(seq_len, device=x.device)
            if key_len == seq_len and past_len == 0
            else causal_mask_with_offset(
                seq_len,
                key_len,
                device=x.device,
                query_offset=past_len,
            )
        )
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        out = self.dense(out)
        out = out * query_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out, (k, v, attention_mask)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out, _ = self.forward_with_cache(x, attention_mask)
        return out


class BloomBlock(nn.Module):
    def __init__(self, config: BloomConfig) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate = (
            int(config.intermediate_size)
            if config.intermediate_size is not None
            else hidden_size * 4
        )
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=float(config.layer_norm_eps))
        self.self_attention = BloomAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=float(config.layer_norm_eps))
        self.mlp = GELUMLP(hidden_size, intermediate, dropout=float(config.dropout))
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attention(self.input_layernorm(x), attention_mask))
        x = x + self.dropout(self.mlp(self.post_attention_layernorm(x)))
        return x

    def forward_with_cache(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        attn_out, next_cache = self.self_attention.forward_with_cache(
            self.input_layernorm(x),
            attention_mask,
            past_key_value=past_key_value,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.post_attention_layernorm(x)))
        return x, next_cache


class BloomModel(nn.Module):
    def __init__(self, config: BloomConfig) -> None:
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(int(config.vocab_size), int(config.hidden_size))
        self.word_embeddings_layernorm = nn.LayerNorm(
            int(config.hidden_size), eps=float(config.layer_norm_eps)
        )
        self.h = nn.ModuleList([BloomBlock(config) for _ in range(int(config.num_layers))])
        self.ln_f = nn.LayerNorm(int(config.hidden_size), eps=float(config.layer_norm_eps))
        self.lm_head = nn.Linear(int(config.hidden_size), int(config.vocab_size), bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = make_attention_mask(input_ids, attention_mask)
        x = self.word_embeddings(input_ids.to(torch.long))
        x = self.word_embeddings_layernorm(x)
        for block in self.h:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.lm_head(x)

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]]:
        mask = make_attention_mask(input_ids, attention_mask)
        x = self.word_embeddings(input_ids.to(torch.long))
        x = self.word_embeddings_layernorm(x)
        if past_key_values is None:
            layer_past = [None] * len(self.h)
        else:
            if len(past_key_values) != len(self.h):
                raise ValueError("past_key_values must match number of layers")
            layer_past = list(past_key_values)
        next_cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for idx, block in enumerate(self.h):
            x, layer_cache = block.forward_with_cache(
                x,
                mask,
                past_key_value=layer_past[idx],
            )
            next_cache.append(layer_cache)
        x = self.ln_f(x)
        return self.lm_head(x), tuple(next_cache)


__all__ = ["BloomAttention", "BloomBlock", "BloomConfig", "BloomModel"]
