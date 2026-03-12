from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .blip2 import BLIP2QFormer
from .llama import LLaMAConfig, LLaMAModel


def format_instructblip_prompt(*, instruction: str, question: str) -> str:
    return f"Instruction: {str(instruction).strip()}\nQuestion: {str(question).strip()}\nAnswer:"


class InstructionAwareQFormer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.instruction_attn = nn.MultiheadAttention(int(hidden_size), int(num_heads), batch_first=True)
        self.instruction_norm = nn.LayerNorm(int(hidden_size))
        self.base_q_former = BLIP2QFormer(int(hidden_size), int(num_heads), int(num_layers))

    def forward(
        self,
        queries: torch.Tensor,
        image_tokens: torch.Tensor,
        instruction_tokens: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.instruction_attn(
            queries,
            instruction_tokens,
            instruction_tokens,
            need_weights=False,
        )
        queries = self.instruction_norm(queries + attended)
        return self.base_q_former(queries, image_tokens)


@dataclass(frozen=True)
class InstructBLIPConfig:
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


class InstructBLIPModel(nn.Module):
    instruction_tuning_enabled = True
    llm_frozen = True
    base_variant = "blip2"

    def __init__(self, config: InstructBLIPConfig) -> None:
        super().__init__()
        self.config = config
        self.image_encoder = nn.Linear(int(config.image_feat_dim), int(config.qformer_hidden_size))
        self.instruction_embeddings = nn.Embedding(
            int(config.vocab_size),
            int(config.qformer_hidden_size),
        )
        self.query_tokens = nn.Parameter(
            torch.randn(int(config.num_query_tokens), int(config.qformer_hidden_size)) * 0.02
        )
        self.q_former = InstructionAwareQFormer(
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

    def encode_image_with_instruction(
        self,
        *,
        image_features: torch.Tensor,
        instruction_ids: torch.Tensor,
    ) -> torch.Tensor:
        image_tokens = self.image_encoder(image_features)
        instruction_tokens = self.instruction_embeddings(instruction_ids.to(torch.long))
        queries = self.query_tokens.unsqueeze(0).expand(image_features.shape[0], -1, -1)
        return self.q_former(queries, image_tokens, instruction_tokens)

    def forward(
        self,
        *,
        image_features: torch.Tensor,
        instruction_ids: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        visual_queries = self.encode_image_with_instruction(
            image_features=image_features,
            instruction_ids=instruction_ids,
        )
        prompts = self.llm_projection(visual_queries)
        prompt_dict = {idx: prompts for idx in range(len(self.llm.layers))}
        gate = torch.ones(int(self.llm.config.n_heads), device=input_ids.device)
        gate_dict = {idx: gate for idx in range(len(self.llm.layers))}
        hidden = self.llm.forward_hidden(input_ids, prompts=prompt_dict, prompt_gates=gate_dict)
        return self.llm.lm_head(hidden)


__all__ = [
    "InstructBLIPConfig",
    "InstructBLIPModel",
    "InstructionAwareQFormer",
    "format_instructblip_prompt",
]
