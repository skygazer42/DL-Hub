from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .t5 import T5Config, T5Model


def format_instruction_prompt(instruction: str, input_text: str) -> str:
    inst = str(instruction).strip()
    text = str(input_text).strip()
    if not inst:
        return text
    if not text:
        return inst
    return f"{inst}\n{text}"


@dataclass(frozen=True)
class FlanT5Config:
    vocab_size: int
    max_seq_len: int
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.0
    instruction_prefix: str = ""

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


class FlanT5Model(nn.Module):
    def __init__(self, config: FlanT5Config) -> None:
        super().__init__()
        self.config = config
        self.base_model = T5Model(config.to_t5_config())

    def build_prompt(self, instruction: str, input_text: str) -> str:
        prefix = self.config.instruction_prefix or instruction
        return format_instruction_prompt(prefix, input_text)

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


__all__ = ["FlanT5Config", "FlanT5Model", "format_instruction_prompt"]
