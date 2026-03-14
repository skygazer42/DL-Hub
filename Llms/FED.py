from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ._shared import GELUMLP


@dataclass(frozen=True)
class FEDConfig:
    vocab_size: int
    max_seq_len: int
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 6
    intermediate_size: int | None = None
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5


class FEDMultiQueryAttention(nn.Module):
    multi_query = True

    def __init__(self, config: FEDConfig) -> None:
        super().__init__()
        if int(config.hidden_size) % int(config.num_heads) != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_heads)
        self.head_dim = int(config.hidden_size // config.num_heads)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(float(config.dropout))

    def cache_elements_per_token(self) -> int:
        return 2 * self.head_dim

    def standard_mha_cache_elements_per_token(self) -> int:
        return 2 * self.hidden_size

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        new_k = self.k_proj(x)
        new_v = self.v_proj(x)

        past_length = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_length = int(past_k.shape[1])
            k = torch.cat((past_k, new_k), dim=1)
            v = torch.cat((past_v, new_v), dim=1)
        else:
            k = new_k
            v = new_v

        total_len = int(k.shape[1])
        expanded_k = k.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        expanded_v = v.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        scores = torch.matmul(q, expanded_k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        query_positions = torch.arange(
            past_length,
            past_length + seq_len,
            device=x.device,
        ).view(1, 1, seq_len, 1)
        key_positions = torch.arange(total_len, device=x.device).view(1, 1, 1, total_len)
        causal = key_positions <= query_positions
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        out = torch.matmul(probs, expanded_v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        out = self.o_proj(out)
        present = (k, v) if use_cache else None
        return out, present


class FEDDecoderBlock(nn.Module):
    def __init__(self, config: FEDConfig) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = (
            int(config.intermediate_size)
            if config.intermediate_size is not None
            else hidden_size * 4
        )
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=float(config.layer_norm_eps))
        self.attention = FEDMultiQueryAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=float(config.layer_norm_eps))
        self.mlp = GELUMLP(hidden_size, intermediate_size, dropout=float(config.dropout))
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, present = self.attention(
            self.input_layernorm(x),
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.post_attention_layernorm(x)))
        return x, present


class FEDModel(nn.Module):
    def __init__(self, config: FEDConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(int(config.vocab_size), int(config.hidden_size))
        self.position_embeddings = nn.Embedding(int(config.max_seq_len), int(config.hidden_size))
        self.layers = nn.ModuleList([FEDDecoderBlock(config) for _ in range(int(config.num_layers))])
        self.norm = nn.LayerNorm(int(config.hidden_size), eps=float(config.layer_norm_eps))
        self.lm_head = nn.Linear(int(config.hidden_size), int(config.vocab_size), bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        ids = input_ids.to(torch.long)
        past_length = 0 if past_key_values is None else int(past_key_values[0][0].shape[1])
        position_ids = torch.arange(
            past_length,
            past_length + ids.shape[1],
            device=ids.device,
        )
        x = self.embed_tokens(ids) + self.position_embeddings(position_ids).unsqueeze(0)

        presents: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[layer_index]
            x, present = layer(x, past_key_value=layer_past, use_cache=use_cache)
            if use_cache and present is not None:
                presents.append(present)
        x = self.norm(x)
        if use_cache:
            return x, tuple(presents)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        hidden = self.forward_hidden(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if use_cache:
            hidden_states, presents = hidden
            return self.lm_head(hidden_states), presents
        return self.lm_head(hidden)


__all__ = [
    "FEDConfig",
    "FEDDecoderBlock",
    "FEDModel",
    "FEDMultiQueryAttention",
]
