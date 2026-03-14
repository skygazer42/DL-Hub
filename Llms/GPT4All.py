from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .llama import LLaMAConfig, LLaMAModel


def format_gpt4all_prompt(user_prompt: str) -> str:
    prompt = str(user_prompt).strip()
    return f"### User:\n{prompt}\n\n### Assistant:\n"


@dataclass(frozen=True)
class GPT4AllDataCuration:
    initial_pairs: int = 1_000_000
    cleaned_pairs: int = 806_199
    final_pairs: int = 437_605
    distillation_source: str = "gpt-3.5-turbo"
    removed_subsets: tuple[str, ...] = ("Bigscience/P3",)


@dataclass(frozen=True)
class GPT4AllConfig:
    vocab_size: int
    max_seq_len: int
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 6
    intermediate_size: int | None = None
    dropout: float = 0.0
    lora_rank: int = 8
    quantization_bits: int = 4

    def to_llama_config(self) -> LLaMAConfig:
        return LLaMAConfig(
            vocab_size=int(self.vocab_size),
            max_seq_len=int(self.max_seq_len),
            dim=int(self.hidden_size),
            n_heads=int(self.num_heads),
            n_layers=int(self.num_layers),
            intermediate_size=self.intermediate_size,
            dropout=float(self.dropout),
        )


class GPT4AllModel(nn.Module):
    training_strategy = "lora"

    def __init__(self, config: GPT4AllConfig) -> None:
        super().__init__()
        self.config = config
        self.data_curation = GPT4AllDataCuration()
        self.quantization_bits = int(config.quantization_bits)
        self.base_model = LLaMAModel(config.to_llama_config())

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.base_model(input_ids, attention_mask)


__all__ = [
    "GPT4AllConfig",
    "GPT4AllDataCuration",
    "GPT4AllModel",
    "format_gpt4all_prompt",
]
