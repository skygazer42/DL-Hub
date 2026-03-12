from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .llama import LLaMAConfig, LLaMAModel


class BLIP2QFormerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, *, with_cross_attention: bool) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(hidden_size)
        self.with_cross_attention = bool(with_cross_attention)
        if self.with_cross_attention:
            self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
            self.cross_norm = nn.LayerNorm(hidden_size)
        else:
            self.cross_attn = None
            self.cross_norm = None
        self.ff = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, queries: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        x, _ = self.self_attn(queries, queries, queries, need_weights=False)
        queries = queries + x
        queries = self.self_norm(queries)
        if self.with_cross_attention and self.cross_attn is not None and self.cross_norm is not None:
            x, _ = self.cross_attn(queries, image_tokens, image_tokens, need_weights=False)
            queries = self.cross_norm(queries + x)
        return queries + self.ff(queries)


class BLIP2QFormer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BLIP2QFormerBlock(hidden_size, num_heads, with_cross_attention=(idx % 2 == 0))
                for idx in range(int(num_layers))
            ]
        )

    def forward(self, queries: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        x = queries
        for layer in self.layers:
            x = layer(x, image_tokens)
        return x


@dataclass(frozen=True)
class BLIP2Config:
    vocab_size: int
    max_seq_len: int
    llm_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    intermediate_size: int = 2048
    image_feat_dim: int = 256
    num_query_tokens: int = 32
    qformer_hidden_size: int = 256
    dropout: float = 0.0


class BLIP2Model(nn.Module):
    def __init__(self, config: BLIP2Config) -> None:
        super().__init__()
        self.config = config
        self.image_encoder = nn.Linear(int(config.image_feat_dim), int(config.qformer_hidden_size))
        self.query_tokens = nn.Parameter(
            torch.randn(int(config.num_query_tokens), int(config.qformer_hidden_size)) * 0.02
        )
        self.q_former = BLIP2QFormer(
            int(config.qformer_hidden_size),
            int(config.num_heads),
            max(1, int(config.num_layers)),
        )
        self.llm_projection = nn.Linear(int(config.qformer_hidden_size), int(config.llm_dim))
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

        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.llm.parameters():
            param.requires_grad = False

    def encode_image(self, image_features: torch.Tensor) -> torch.Tensor:
        image_tokens = self.image_encoder(image_features)
        queries = self.query_tokens.unsqueeze(0).expand(image_features.shape[0], -1, -1)
        return self.q_former(queries, image_tokens)

    def forward(self, *, image_features: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        visual_queries = self.encode_image(image_features)
        prompts = self.llm_projection(visual_queries)
        prompt_dict = {idx: prompts for idx in range(len(self.llm.layers))}
        gate = torch.ones(int(self.llm.config.n_heads), device=input_ids.device)
        gate_dict = {idx: gate for idx in range(len(self.llm.layers))}
        hidden = self.llm.forward_hidden(input_ids, prompts=prompt_dict, prompt_gates=gate_dict)
        return self.llm.lm_head(hidden)


__all__ = ["BLIP2Config", "BLIP2Model", "BLIP2QFormer", "BLIP2QFormerBlock"]
