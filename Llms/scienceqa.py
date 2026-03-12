from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def format_scienceqa_prompt(example: "ScienceQAExample") -> str:
    choices = "\n".join(
        f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(example.choices)
    )
    context_lines = []
    if example.text_context:
        context_lines.append(f"Context: {example.text_context}")
    if example.lecture:
        context_lines.append(f"Lecture: {example.lecture}")
    if example.has_image:
        context_lines.append("Image: available")
    context = "\n".join(context_lines)
    return (
        f"{context}\nQuestion: {example.question}\nChoices:\n{choices}\n"
        "Let's think step by step."
    ).strip()


@dataclass(frozen=True)
class ScienceQAExample:
    question: str
    choices: tuple[str, ...]
    answer_index: int
    lecture: str = ""
    explanation: str = ""
    text_context: str = ""
    has_image: bool = False


@dataclass(frozen=True)
class ScienceQAConfig:
    vocab_size: int
    max_seq_len: int
    hidden_size: int = 256
    image_feature_dim: int = 128
    num_choices: int = 4
    dropout: float = 0.0


class ScienceQAModel(nn.Module):
    cot_enabled = True
    supports_multimodal_context = True

    def __init__(self, config: ScienceQAConfig) -> None:
        super().__init__()
        self.config = config
        self.text_embeddings = nn.Embedding(int(config.vocab_size), int(config.hidden_size))
        self.image_projector = nn.Linear(int(config.image_feature_dim), int(config.hidden_size))
        self.encoder = nn.GRU(
            input_size=int(config.hidden_size),
            hidden_size=int(config.hidden_size),
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(config.dropout))
        self.answer_head = nn.Linear(int(config.hidden_size), int(config.num_choices))

    def forward(
        self,
        *,
        question_ids: torch.Tensor,
        image_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.text_embeddings(question_ids.to(torch.long))
        encoded, _ = self.encoder(hidden)
        pooled = encoded[:, -1]
        if image_features is not None:
            pooled = pooled + self.image_projector(image_features.to(dtype=pooled.dtype))
        pooled = self.dropout(pooled)
        return self.answer_head(pooled)


__all__ = [
    "ScienceQAConfig",
    "ScienceQAExample",
    "ScienceQAModel",
    "format_scienceqa_prompt",
]
