from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ._shared import (
    RMSNorm,
    apply_rotary_embeddings,
    causal_mask,
    causal_mask_with_offset,
    expand_key_padding_mask,
    make_attention_mask,
    SwiGLUMLP,
)


@dataclass(frozen=True)
class LLaMAConfig:
    vocab_size: int
    max_seq_len: int
    dim: int = 512
    n_heads: int = 8
    n_layers: int = 6
    intermediate_size: int | None = None
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0


class LLaMAAttention(nn.Module):
    use_rope = True

    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        if int(config.dim) % int(config.n_heads) != 0:
            raise ValueError("dim must be divisible by n_heads")
        self.hidden_size = int(config.dim)
        self.num_heads = int(config.n_heads)
        self.head_dim = int(config.dim // config.n_heads)
        self.rotary_ndims = self.head_dim
        self.rope_theta = float(config.rope_theta)

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(float(config.dropout))

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _token_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        query_mask: torch.Tensor,
        query_offset: int = 0,
    ) -> torch.Tensor:
        bsz, _, seq_len, _ = q.shape
        key_len = int(k.shape[-2])
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(
            ~expand_key_padding_mask(attention_mask, batch_size=bsz, seq_len=key_len),
            torch.finfo(scores.dtype).min,
        )
        causal = (
            causal_mask(seq_len, device=q.device)
            if key_len == seq_len and int(query_offset) == 0
            else causal_mask_with_offset(
                seq_len,
                key_len,
                device=q.device,
                query_offset=int(query_offset),
            )
        )
        scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        out = torch.matmul(probs, v)
        return out * query_mask.view(bsz, 1, seq_len, 1).to(dtype=out.dtype)

    def forward_with_cache(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        prompt: torch.Tensor | None = None,
        prompt_gate: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        bsz, seq_len, _ = x.shape
        past_len = 0
        query_mask = attention_mask
        q = self._reshape(self.q_proj(x))
        k = self._reshape(self.k_proj(x))
        v = self._reshape(self.v_proj(x))
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
        out = self._token_attention(
            q,
            k,
            v,
            attention_mask,
            query_mask=query_mask,
            query_offset=past_len,
        )

        if prompt is not None:
            prompt = prompt.to(dtype=x.dtype, device=x.device)
            if prompt.ndim != 3 or prompt.shape[0] != bsz or prompt.shape[2] != self.hidden_size:
                raise ValueError(
                    f"prompt must be (B, K, {self.hidden_size}), got {tuple(prompt.shape)}"
                )
            prompt_k = self._reshape(self.k_proj(prompt))
            prompt_v = self._reshape(self.v_proj(prompt))
            prompt_scores = torch.matmul(q, prompt_k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            prompt_probs = torch.softmax(prompt_scores, dim=-1)
            if prompt_gate is None:
                raise ValueError("prompt_gate is required when prompt is provided")
            prompt_probs = prompt_probs * prompt_gate.view(1, self.num_heads, 1, 1)
            out = out + torch.matmul(prompt_probs, prompt_v)

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        out = self.o_proj(out)
        out = out * query_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out, (k, v, attention_mask)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        prompt: torch.Tensor | None = None,
        prompt_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out, _ = self.forward_with_cache(
            x,
            attention_mask,
            prompt=prompt,
            prompt_gate=prompt_gate,
        )
        return out


class LLaMABlock(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        hidden_size = int(config.dim)
        intermediate = (
            int(config.intermediate_size)
            if config.intermediate_size is not None
            else hidden_size * 4
        )
        self.attention_norm = RMSNorm(hidden_size, eps=float(config.rms_norm_eps))
        self.attention = LLaMAAttention(config)
        self.ffn_norm = RMSNorm(hidden_size, eps=float(config.rms_norm_eps))
        self.feed_forward = SwiGLUMLP(hidden_size, intermediate, dropout=float(config.dropout))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        prompt: torch.Tensor | None = None,
        prompt_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.attention_norm(x),
            attention_mask,
            prompt=prompt,
            prompt_gate=prompt_gate,
        )
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def forward_with_cache(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        prompt: torch.Tensor | None = None,
        prompt_gate: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        attn_out, next_cache = self.attention.forward_with_cache(
            self.attention_norm(x),
            attention_mask,
            prompt=prompt,
            prompt_gate=prompt_gate,
            past_key_value=past_key_value,
        )
        x = x + attn_out
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, next_cache


class LLaMAModel(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(int(config.vocab_size), int(config.dim))
        self.layers = nn.ModuleList([LLaMABlock(config) for _ in range(int(config.n_layers))])
        self.norm = RMSNorm(int(config.dim), eps=float(config.rms_norm_eps))
        self.lm_head = nn.Linear(int(config.dim), int(config.vocab_size), bias=False)

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        prompts: dict[int, torch.Tensor] | None = None,
        prompt_gates: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        mask = make_attention_mask(input_ids, attention_mask)
        x = self.tok_embeddings(input_ids.to(torch.long))
        prompts = {} if prompts is None else prompts
        prompt_gates = {} if prompt_gates is None else prompt_gates
        for idx, layer in enumerate(self.layers):
            x = layer(
                x,
                mask,
                prompt=prompts.get(idx),
                prompt_gate=prompt_gates.get(idx),
            )
        return self.norm(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.forward_hidden(input_ids, attention_mask)
        return self.lm_head(hidden)

    def forward_hidden_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None, ...] | None = None,
        prompts: dict[int, torch.Tensor] | None = None,
        prompt_gates: dict[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]]:
        mask = make_attention_mask(input_ids, attention_mask)
        x = self.tok_embeddings(input_ids.to(torch.long))
        prompts = {} if prompts is None else prompts
        prompt_gates = {} if prompt_gates is None else prompt_gates
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
                prompt=prompts.get(idx),
                prompt_gate=prompt_gates.get(idx),
                past_key_value=layer_past[idx],
            )
            next_cache.append(layer_cache)
        return self.norm(x), tuple(next_cache)

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None, ...] | None = None,
        prompts: dict[int, torch.Tensor] | None = None,
        prompt_gates: dict[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]]:
        hidden, next_cache = self.forward_hidden_with_cache(
            input_ids,
            attention_mask,
            past_key_values=past_key_values,
            prompts=prompts,
            prompt_gates=prompt_gates,
        )
        return self.lm_head(hidden), next_cache


__all__ = ["LLaMAAttention", "LLaMABlock", "LLaMAConfig", "LLaMAModel"]
