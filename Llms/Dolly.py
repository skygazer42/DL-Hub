from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .pythia import PythiaConfig, PythiaModel


def format_dolly_prompt(instruction: str, context: str = "") -> str:
    inst = str(instruction).strip()
    ctx = str(context).strip()
    sections = [f"### Instruction:\n{inst}"]
    if ctx:
        sections.append(f"### Input:\n{ctx}")
    else:
        sections.append("### Input:\n")
    sections.append("### Response:\n")
    return "\n\n".join(sections)


@dataclass(frozen=True)
class DollyExample:
    instruction: str
    context: str
    response: str
    category: str = "open_qa"


@dataclass(frozen=True)
class DollyConfig:
    vocab_size: int
    max_seq_len: int
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 6
    intermediate_size: int | None = None
    rotary_pct: float = 0.25
    dropout: float = 0.0

    def to_pythia_config(self) -> PythiaConfig:
        return PythiaConfig(
            vocab_size=int(self.vocab_size),
            max_seq_len=int(self.max_seq_len),
            hidden_size=int(self.hidden_size),
            num_heads=int(self.num_heads),
            num_layers=int(self.num_layers),
            intermediate_size=self.intermediate_size,
            rotary_pct=float(self.rotary_pct),
            dropout=float(self.dropout),
            deduped=True,
        )


class DollyModel(nn.Module):
    dataset_name = "databricks-dolly-15k"
    human_generated_data = True

    def __init__(self, config: DollyConfig) -> None:
        super().__init__()
        self.config = config
        self.base_model = PythiaModel(config.to_pythia_config())

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask)


__all__ = [
    "DollyConfig",
    "DollyExample",
    "DollyModel",
    "format_dolly_prompt",
]
