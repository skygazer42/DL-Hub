from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .blip2 import BLIP2QFormer
from .llama import LLaMAConfig, LLaMAModel


def format_minigpt4_prompt(instruction: str) -> str:
    text = str(instruction).strip()
    return f"###Human: <Img><ImageFeature></Img> {text} ###Assistant:"


@dataclass(frozen=True)
class MiniGPT4Config:
    vocab_size: int
    max_seq_len: int
    llm_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    intermediate_size: int = 2048
    image_feat_dim: int = 256
    visual_token_count: int = 4
    qformer_hidden_size: int | None = None
    dropout: float = 0.0


class MiniGPT4Model(nn.Module):
    def __init__(self, config: MiniGPT4Config) -> None:
        super().__init__()
        self.config = config
        hidden = (
            int(config.qformer_hidden_size)
            if config.qformer_hidden_size is not None
            else int(config.image_feat_dim)
        )
        self.vision_encoder = nn.Linear(int(config.image_feat_dim), hidden)
        self.query_tokens = nn.Parameter(torch.randn(int(config.visual_token_count), hidden) * 0.02)
        self.q_former = BLIP2QFormer(hidden, int(config.num_heads), max(1, int(config.num_layers)))
        self.vision_projection = nn.Linear(hidden, int(config.llm_dim))
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

        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.q_former.parameters():
            param.requires_grad = False
        self.query_tokens.requires_grad = False
        for param in self.llm.parameters():
            param.requires_grad = False

    def encode_image(self, image_features: torch.Tensor) -> torch.Tensor:
        image_tokens = self.vision_encoder(image_features)
        queries = self.query_tokens.unsqueeze(0).expand(image_features.shape[0], -1, -1)
        return self.q_former(queries, image_tokens)

    def forward(self, *, image_features: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        visual_tokens = self.vision_projection(self.encode_image(image_features))
        prompt_dict = {idx: visual_tokens for idx in range(len(self.llm.layers))}
        gate = torch.ones(int(self.llm.config.n_heads), device=input_ids.device)
        gate_dict = {idx: gate for idx in range(len(self.llm.layers))}
        hidden = self.llm.forward_hidden(input_ids, prompts=prompt_dict, prompt_gates=gate_dict)
        return self.llm.lm_head(hidden)


__all__ = ["MiniGPT4Config", "MiniGPT4Model", "format_minigpt4_prompt"]
