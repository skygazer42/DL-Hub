from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .llama import LLaMAConfig, LLaMAModel
from ._shared import make_attention_mask


class PerceiverResampler(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_latents: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
        self.latents = nn.Parameter(torch.randn(int(num_latents), int(hidden_dim)) * 0.02)
        self.cross_attn = nn.MultiheadAttention(int(hidden_dim), int(num_heads), batch_first=True)
        self.norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        tokens = self.input_proj(image_features)
        latents = self.latents.unsqueeze(0).expand(image_features.shape[0], -1, -1)
        out, _ = self.cross_attn(latents, tokens, tokens, need_weights=False)
        return self.norm(latents + out)


class GatedCrossAttentionDenseBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(int(hidden_dim))
        self.cross_attn = nn.MultiheadAttention(int(hidden_dim), int(num_heads), batch_first=True)
        self.ff_norm = nn.LayerNorm(int(hidden_dim))
        self.ff = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim) * 4),
            nn.GELU(),
            nn.Linear(int(hidden_dim) * 4, int(hidden_dim)),
        )
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.ff_gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        attn_in = self.attn_norm(hidden_states)
        attn_out, _ = self.cross_attn(attn_in, visual_tokens, visual_tokens, need_weights=False)
        hidden_states = hidden_states + torch.tanh(self.attn_gate) * attn_out
        ff_out = self.ff(self.ff_norm(hidden_states))
        hidden_states = hidden_states + torch.tanh(self.ff_gate) * ff_out
        return hidden_states


@dataclass(frozen=True)
class FlamingoConfig:
    vocab_size: int
    max_seq_len: int
    llm_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    intermediate_size: int = 2048
    image_feat_dim: int = 256
    resampler_num_latents: int = 64
    cross_attn_every_n_layers: int = 4
    dropout: float = 0.0


class FlamingoModel(nn.Module):
    def __init__(self, config: FlamingoConfig) -> None:
        super().__init__()
        self.config = config
        self.llm = LLaMAModel(
            LLaMAConfig(
                vocab_size=int(config.vocab_size),
                max_seq_len=int(config.max_seq_len),
                dim=int(config.llm_dim),
                n_heads=int(config.num_heads),
                n_layers=int(config.num_layers),
                intermediate_size=int(config.intermediate_size),
                dropout=float(config.dropout),
            )
        )
        for param in self.llm.parameters():
            param.requires_grad = False

        self.perceiver_resampler = PerceiverResampler(
            int(config.image_feat_dim),
            int(config.llm_dim),
            int(config.num_heads),
            int(config.resampler_num_latents),
        )
        every = max(1, int(config.cross_attn_every_n_layers))
        self.cross_attn_layer_indices = list(range(0, int(config.num_layers), every))
        self.gated_xattn_layers = nn.ModuleList(
            [
                GatedCrossAttentionDenseBlock(int(config.llm_dim), int(config.num_heads))
                for _ in self.cross_attn_layer_indices
            ]
        )

    def forward(self, *, image_features: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        visual_tokens = self.perceiver_resampler(image_features)
        mask = make_attention_mask(input_ids)
        x = self.llm.tok_embeddings(input_ids.to(torch.long))
        gate_map = {
            layer_idx: self.gated_xattn_layers[idx]
            for idx, layer_idx in enumerate(self.cross_attn_layer_indices)
        }
        for idx, layer in enumerate(self.llm.layers):
            if idx in gate_map:
                x = gate_map[idx](x, visual_tokens)
            x = layer(x, mask)
        x = self.llm.norm(x)
        return self.llm.lm_head(x)


__all__ = [
    "FlamingoConfig",
    "FlamingoModel",
    "GatedCrossAttentionDenseBlock",
    "PerceiverResampler",
]
