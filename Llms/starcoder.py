from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .gpt_neox import GPTNeoXConfig, GPTNeoXModel


def format_fim_prompt(*, prefix: str, suffix: str) -> str:
    return f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"


@dataclass(frozen=True)
class StarCoderDataConfig:
    num_languages: int = 80
    context_window: int = 8192
    license: str = "OpenRAIL"
    training_sources: tuple[str, ...] = (
        "GitHub repositories",
        "Git commits",
        "GitHub issues",
        "Jupyter notebooks",
    )
    pii_redaction_enabled: bool = True


@dataclass(frozen=True)
class StarCoderConfig:
    vocab_size: int
    max_seq_len: int
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 6
    intermediate_size: int | None = None
    dropout: float = 0.0

    def to_gpt_neox_config(self) -> GPTNeoXConfig:
        return GPTNeoXConfig(
            vocab_size=int(self.vocab_size),
            max_seq_len=int(self.max_seq_len),
            hidden_size=int(self.hidden_size),
            num_heads=int(self.num_heads),
            num_layers=int(self.num_layers),
            intermediate_size=self.intermediate_size,
            rotary_pct=0.25,
            dropout=float(self.dropout),
        )


class StarCoderModel(nn.Module):
    supports_fill_in_the_middle = True
    assistant_capabilities = ("code-completion", "code-editing", "code-explanation")

    def __init__(self, config: StarCoderConfig) -> None:
        super().__init__()
        self.config = config
        self.data_config = StarCoderDataConfig()
        self.base_model = GPTNeoXModel(config.to_gpt_neox_config())

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.base_model(input_ids, attention_mask)


__all__ = [
    "StarCoderConfig",
    "StarCoderDataConfig",
    "StarCoderModel",
    "format_fim_prompt",
]
