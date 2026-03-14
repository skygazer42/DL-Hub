from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ._shared import (
    apply_rotary_embeddings,
    causal_mask,
    causal_mask_with_offset,
    expand_key_padding_mask,
    GELUMLP,
    make_attention_mask,
)


@dataclass(frozen=True)
class GPTNeoXConfig:
    vocab_size: int
    max_seq_len: int
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 6
    intermediate_size: int | None = None
    rotary_pct: float = 0.25
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0


class GPTNeoXAttention(nn.Module):
    def __init__(self, config: GPTNeoXConfig) -> None:
        super().__init__()
        if int(config.hidden_size) % int(config.num_heads) != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_heads)
        self.head_dim = int(config.hidden_size // config.num_heads)
        self.rotary_ndims = max(2, int(self.head_dim * float(config.rotary_pct)))
        if self.rotary_ndims % 2 == 1:
            self.rotary_ndims -= 1
        self.rotary_ndims = max(2, min(self.rotary_ndims, self.head_dim))
        self.rope_theta = float(config.rope_theta)

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
        query_mask = attention_mask
        qkv = self.query_key_value(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        if past_key_value is not None:
            past_len = int(past_key_value[0].shape[-2])
        q, k = apply_rotary_embeddings(
            q,
            k,
            rotary_dim=self.rotary_ndims,
            base=self.rope_theta,
            position_offset=past_len,
        )
        if past_key_value is not None:
            past_k, past_v, past_mask = past_key_value
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


class GPTNeoXParallelBlock(nn.Module):
    parallel_residual = True

    def __init__(self, config: GPTNeoXConfig) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate = (
            int(config.intermediate_size)
            if config.intermediate_size is not None
            else hidden_size * 4
        )
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=float(config.layer_norm_eps))
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=float(config.layer_norm_eps))
        self.attention = GPTNeoXAttention(config)
        self.mlp = GELUMLP(hidden_size, intermediate, dropout=float(config.dropout))
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attn_input = self.input_layernorm(x)
        mlp_input = self.post_attention_layernorm(x)
        attn_out = self.dropout(self.attention(attn_input, attention_mask))
        mlp_out = self.dropout(self.mlp(mlp_input))
        return x + attn_out + mlp_out

    def forward_with_cache(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        attn_input = self.input_layernorm(x)
        mlp_input = self.post_attention_layernorm(x)
        attn_out, next_cache = self.attention.forward_with_cache(
            attn_input,
            attention_mask,
            past_key_value=past_key_value,
        )
        attn_out = self.dropout(attn_out)
        mlp_out = self.dropout(self.mlp(mlp_input))
        return x + attn_out + mlp_out, next_cache


class GPTNeoXModel(nn.Module):
    def __init__(self, config: GPTNeoXConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_in = nn.Embedding(int(config.vocab_size), int(config.hidden_size))
        self.layers = nn.ModuleList(
            [GPTNeoXParallelBlock(config) for _ in range(int(config.num_layers))]
        )
        self.final_layer_norm = nn.LayerNorm(
            int(config.hidden_size), eps=float(config.layer_norm_eps)
        )
        self.embed_out = nn.Linear(int(config.hidden_size), int(config.vocab_size), bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = make_attention_mask(input_ids, attention_mask)
        x = self.embed_in(input_ids.to(torch.long))
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_layer_norm(x)
        return self.embed_out(x)

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]]:
        mask = make_attention_mask(input_ids, attention_mask)
        x = self.embed_in(input_ids.to(torch.long))
        if past_key_values is None:
            layer_past = [None] * len(self.layers)
        else:
            if len(past_key_values) != len(self.layers):
                raise ValueError("past_key_values must match number of layers")
            layer_past = list(past_key_values)
        next_cache: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for idx, layer in enumerate(self.layers):
            x, layer_cache = layer.forward_with_cache(
                x,
                mask,
                past_key_value=layer_past[idx],
            )
            next_cache.append(layer_cache)
        x = self.final_layer_norm(x)
        return self.embed_out(x), tuple(next_cache)


__all__ = ["GPTNeoXAttention", "GPTNeoXConfig", "GPTNeoXModel", "GPTNeoXParallelBlock"]
