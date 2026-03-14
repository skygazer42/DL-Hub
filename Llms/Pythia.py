from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .gpt_neox import GPTNeoXConfig, GPTNeoXModel


@dataclass(frozen=True)
class PythiaCheckpointSchedule:
    total_train_steps: int = 143000
    checkpoint_interval: int = 1000
    early_steps: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

    @property
    def steps(self) -> list[int]:
        dense_steps = list(range(int(self.checkpoint_interval), int(self.total_train_steps) + 1, int(self.checkpoint_interval)))
        all_steps = sorted({*self.early_steps, *dense_steps})
        return [step for step in all_steps if step <= int(self.total_train_steps)]


@dataclass(frozen=True)
class PythiaConfig:
    vocab_size: int
    max_seq_len: int
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 6
    intermediate_size: int | None = None
    rotary_pct: float = 0.25
    dropout: float = 0.0
    deduped: bool = False
    total_train_steps: int = 143000

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


class PythiaModel(nn.Module):
    uses_flash_attention = True
    data_order_is_reconstructable = True
    suite_variants = ("pythia", "pythia-deduped")

    def __init__(self, config: PythiaConfig) -> None:
        super().__init__()
        self.config = config
        self.checkpoint_schedule = PythiaCheckpointSchedule(total_train_steps=int(config.total_train_steps))
        self.base_model = GPTNeoXModel(config.to_gpt_neox_config())
        self.training_corpus = "The Pile (deduplicated)" if config.deduped else "The Pile"

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.base_model(input_ids, attention_mask)


__all__ = ["PythiaCheckpointSchedule", "PythiaConfig", "PythiaModel"]
