from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .gpt_neox import GPTNeoXConfig, GPTNeoXModel


def build_creative_prompt(*, genre: str, topic: str, style: str) -> str:
    return f"Write a {str(genre).strip()} about {str(topic).strip()} in the style of {str(style).strip()}."


def format_gpt4all_j_prompt(user_prompt: str) -> str:
    prompt = str(user_prompt).strip()
    return f"### Prompt:\n{prompt}\n\n### Response:\n"


@dataclass(frozen=True)
class GPT4AllJDataCuration:
    dataset_points: int = 800_000
    original_dataset_points: int = 400_000
    post_processed_examples: int = 437_605
    base_checkpoint: str = "gpt-j-6.7b"
    license: str = "Apache-2.0"
    creative_domains: tuple[str, ...] = ("poems", "songs", "stories", "plays")
    assistant_sources: tuple[str, ...] = (
        "laion-oig",
        "stackoverflow",
        "bigscience-p3",
        "custom-creative-prompts",
    )


@dataclass(frozen=True)
class GPTJConfig:
    vocab_size: int
    max_seq_len: int
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 6
    intermediate_size: int | None = None
    rotary_pct: float = 0.25
    dropout: float = 0.0

    def to_gpt_neox_config(self) -> GPTNeoXConfig:
        return GPTNeoXConfig(
            vocab_size=int(self.vocab_size),
            max_seq_len=int(self.max_seq_len),
            hidden_size=int(self.hidden_size),
            num_heads=int(self.num_heads),
            num_layers=int(self.num_layers),
            intermediate_size=self.intermediate_size,
            rotary_pct=float(self.rotary_pct),
            dropout=float(self.dropout),
        )


class GPTJBackbone(nn.Module):
    architecture_name = "gpt-j"

    def __init__(self, config: GPTJConfig) -> None:
        super().__init__()
        self.config = config
        self.decoder = GPTNeoXModel(config.to_gpt_neox_config())

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decoder(input_ids, attention_mask)


@dataclass(frozen=True)
class GPT4AllJConfig:
    vocab_size: int
    max_seq_len: int
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 6
    intermediate_size: int | None = None
    dropout: float = 0.0
    quantization_bits: int = 4

    def to_gptj_config(self) -> GPTJConfig:
        return GPTJConfig(
            vocab_size=int(self.vocab_size),
            max_seq_len=int(self.max_seq_len),
            hidden_size=int(self.hidden_size),
            num_heads=int(self.num_heads),
            num_layers=int(self.num_layers),
            intermediate_size=self.intermediate_size,
            dropout=float(self.dropout),
        )


class GPT4AllJModel(nn.Module):
    finetuning_variants = ("full", "lora")

    def __init__(self, config: GPT4AllJConfig) -> None:
        super().__init__()
        self.config = config
        self.data_curation = GPT4AllJDataCuration()
        self.quantization_bits = int(config.quantization_bits)
        self.base_model = GPTJBackbone(config.to_gptj_config())

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask)


__all__ = [
    "GPT4AllJConfig",
    "GPT4AllJDataCuration",
    "GPT4AllJModel",
    "GPTJBackbone",
    "GPTJConfig",
    "build_creative_prompt",
    "format_gpt4all_j_prompt",
]
