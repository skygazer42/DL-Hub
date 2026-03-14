from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ._shared import causal_mask, expand_key_padding_mask, make_attention_mask


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, *, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + self.eps)
        return x_norm.to(dtype=x.dtype) * self.weight


class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        num_heads: int,
        *,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.num_buckets = int(num_buckets)
        self.max_distance = int(max_distance)
        self.bidirectional = bool(bidirectional)
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.num_heads)

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        num_buckets = int(self.num_buckets)
        max_distance = int(self.max_distance)
        ret = torch.zeros_like(relative_position, dtype=torch.long)
        n = -relative_position
        if self.bidirectional:
            num_buckets //= 2
            ret = ret + (n < 0).to(torch.long) * num_buckets
            n = n.abs()
        else:
            n = torch.max(n, torch.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            (
                torch.log(n.to(torch.float32) / max_exact + 1e-6)
                / math.log(max_distance / max_exact)
            )
            * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.minimum(
            val_if_large,
            torch.full_like(val_if_large, num_buckets - 1),
        )
        ret = ret + torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, q_len: int, k_len: int, *, device: torch.device) -> torch.Tensor:
        q_pos = torch.arange(q_len, device=device)[:, None]
        k_pos = torch.arange(k_len, device=device)[None, :]
        rel = k_pos - q_pos
        buckets = self._relative_position_bucket(rel)
        values = self.relative_attention_bias(buckets)
        return values.permute(2, 0, 1).unsqueeze(0)


@dataclass(frozen=True)
class T5Config:
    vocab_size: int
    max_seq_len: int
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128


class T5DenseReluDense(nn.Module):
    activation_name = "relu"

    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.wi = nn.Linear(int(config.d_model), int(config.d_ff), bias=False)
        self.wo = nn.Linear(int(config.d_ff), int(config.d_model), bias=False)
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wo(self.dropout(F.relu(self.wi(x))))


class T5Attention(nn.Module):
    def __init__(
        self,
        config: T5Config,
        *,
        is_decoder: bool,
        has_relative_attention_bias: bool,
    ) -> None:
        super().__init__()
        hidden_size = int(config.d_model)
        num_heads = int(config.num_heads)
        if hidden_size % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.is_decoder = bool(is_decoder)
        self.has_relative_attention_bias = bool(has_relative_attention_bias)
        self.relative_attention_bias = (
            T5RelativePositionBias(
                num_heads,
                num_buckets=int(config.relative_attention_num_buckets),
                max_distance=int(config.relative_attention_max_distance),
                bidirectional=not bool(is_decoder),
            )
            if has_relative_attention_bias
            else None
        )
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(float(config.dropout))

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        key_value_states: torch.Tensor | None = None,
        key_value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        kv_states = hidden_states if key_value_states is None else key_value_states
        kv_mask = attention_mask if key_value_mask is None else key_value_mask
        k_len = int(kv_states.shape[1])

        q = self._shape(self.q(hidden_states))
        k = self._shape(self.k(kv_states))
        v = self._shape(self.v(kv_states))

        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if self.relative_attention_bias is not None:
            scores = scores + self.relative_attention_bias(q_len, k_len, device=hidden_states.device)
        scores = scores.masked_fill(
            ~expand_key_padding_mask(kv_mask, batch_size=bsz, seq_len=k_len),
            torch.finfo(scores.dtype).min,
        )
        if self.is_decoder and key_value_states is None:
            scores = scores.masked_fill(
                ~causal_mask(q_len, device=hidden_states.device),
                torch.finfo(scores.dtype).min,
            )
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        out = self.o(out)
        return out * attention_mask.unsqueeze(-1).to(dtype=out.dtype)


class T5EncoderBlock(nn.Module):
    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.self_attention_layer_norm = T5LayerNorm(
            int(config.d_model), eps=float(config.layer_norm_eps)
        )
        self.self_attention = T5Attention(
            config, is_decoder=False, has_relative_attention_bias=True
        )
        self.ff_layer_norm = T5LayerNorm(int(config.d_model), eps=float(config.layer_norm_eps))
        self.feed_forward = T5DenseReluDense(config)
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attention(self.self_attention_layer_norm(x), attention_mask))
        x = x + self.dropout(self.feed_forward(self.ff_layer_norm(x)))
        return x


class T5DecoderBlock(nn.Module):
    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.self_attention_layer_norm = T5LayerNorm(
            int(config.d_model), eps=float(config.layer_norm_eps)
        )
        self.self_attention = T5Attention(
            config, is_decoder=True, has_relative_attention_bias=True
        )
        self.cross_attention_layer_norm = T5LayerNorm(
            int(config.d_model), eps=float(config.layer_norm_eps)
        )
        self.cross_attention = T5Attention(
            config, is_decoder=False, has_relative_attention_bias=False
        )
        self.ff_layer_norm = T5LayerNorm(int(config.d_model), eps=float(config.layer_norm_eps))
        self.feed_forward = T5DenseReluDense(config)
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(
        self,
        x: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.dropout(
            self.self_attention(self.self_attention_layer_norm(x), decoder_attention_mask)
        )
        x = x + self.dropout(
            self.cross_attention(
                self.cross_attention_layer_norm(x),
                decoder_attention_mask,
                key_value_states=encoder_hidden_states,
                key_value_mask=encoder_attention_mask,
            )
        )
        x = x + self.dropout(self.feed_forward(self.ff_layer_norm(x)))
        return x


class T5Stack(nn.Module):
    def __init__(self, config: T5Config, *, is_decoder: bool, embed_tokens: nn.Embedding) -> None:
        super().__init__()
        self.is_decoder = bool(is_decoder)
        self.embed_tokens = embed_tokens
        block_cls = T5DecoderBlock if is_decoder else T5EncoderBlock
        num_layers = int(config.num_decoder_layers if is_decoder else config.num_encoder_layers)
        self.blocks = nn.ModuleList([block_cls(config) for _ in range(num_layers)])
        self.final_layer_norm = T5LayerNorm(int(config.d_model), eps=float(config.layer_norm_eps))
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.dropout(self.embed_tokens(input_ids.to(torch.long)))
        for block in self.blocks:
            if self.is_decoder:
                if encoder_hidden_states is None or encoder_attention_mask is None:
                    raise ValueError("decoder requires encoder states and mask")
                x = block(
                    x,
                    attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                x = block(x, attention_mask)
        return self.final_layer_norm(x)


class T5Model(nn.Module):
    def __init__(self, config: T5Config) -> None:
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(int(config.vocab_size), int(config.d_model))
        self.encoder = T5Stack(config, is_decoder=False, embed_tokens=self.shared)
        self.decoder = T5Stack(config, is_decoder=True, embed_tokens=self.shared)
        self.lm_head = nn.Linear(int(config.d_model), int(config.vocab_size), bias=False)
        self.lm_head.weight = self.shared.weight

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        enc_mask = make_attention_mask(input_ids, attention_mask)
        dec_mask = make_attention_mask(decoder_input_ids, decoder_attention_mask)
        encoder_hidden = self.encoder(input_ids, enc_mask)
        decoder_hidden = self.decoder(
            decoder_input_ids,
            dec_mask,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=enc_mask,
        )
        return self.lm_head(decoder_hidden)


__all__ = [
    "T5Attention",
    "T5Config",
    "T5DecoderBlock",
    "T5DenseReluDense",
    "T5EncoderBlock",
    "T5LayerNorm",
    "T5Model",
    "T5RelativePositionBias",
    "T5Stack",
]
