from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import nn

from .t5 import T5Config, T5Model


@dataclass(frozen=True)
class RenderedPrompt:
    input_text: str
    target_text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    input_template: str
    target_template: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def materialize(self, example: Mapping[str, Any]) -> RenderedPrompt:
        values = {**dict(self.metadata), **dict(example)}
        return RenderedPrompt(
            input_text=self.input_template.format(**values),
            target_text=self.target_template.format(**values),
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class MTFTask:
    name: str
    templates: tuple[PromptTemplate, ...]
    held_out: bool = False


@dataclass(frozen=True)
class MTFMixture:
    train_tasks: tuple[MTFTask, ...]
    evaluation_tasks: tuple[MTFTask, ...] = ()

    def seen_task_names(self) -> tuple[str, ...]:
        return tuple(task.name for task in self.train_tasks if not task.held_out)

    def zero_shot_task_names(self) -> tuple[str, ...]:
        return tuple(task.name for task in self.evaluation_tasks if task.held_out)


@dataclass(frozen=True)
class MTFConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.0

    def to_t5_config(self) -> T5Config:
        return T5Config(
            vocab_size=int(self.vocab_size),
            max_seq_len=int(self.max_seq_len),
            d_model=int(self.d_model),
            num_heads=int(self.num_heads),
            num_encoder_layers=int(self.num_encoder_layers),
            num_decoder_layers=int(self.num_decoder_layers),
            d_ff=int(self.d_ff),
            dropout=float(self.dropout),
        )


class MTFModel(nn.Module):
    training_objective = "multitask_prompted_training"
    prompt_collection = "PromptSource"
    zero_shot_evaluation = True

    def __init__(self, config: MTFConfig) -> None:
        super().__init__()
        self.config = config
        self.base_model = T5Model(config.to_t5_config())

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.base_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )


__all__ = [
    "MTFConfig",
    "MTFMixture",
    "MTFModel",
    "MTFTask",
    "PromptTemplate",
    "RenderedPrompt",
]
