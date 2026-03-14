from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .llama import LLaMAConfig, LLaMAModel


def _pool_last_hidden(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if attention_mask is None:
        return hidden[:, -1, :]
    last_index = attention_mask.to(torch.long).sum(dim=1).clamp_min(1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last_index]


@dataclass(frozen=True)
class AnthropicConfig:
    vocab_size: int
    max_seq_len: int
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 6
    intermediate_size: int | None = None
    dropout: float = 0.0
    helpfulness_mix: float = 0.5
    harmlessness_mix: float = 0.5

    def to_backbone_config(self) -> LLaMAConfig:
        return LLaMAConfig(
            vocab_size=int(self.vocab_size),
            max_seq_len=int(self.max_seq_len),
            dim=int(self.hidden_size),
            n_heads=int(self.num_heads),
            n_layers=int(self.num_layers),
            intermediate_size=(
                None if self.intermediate_size is None else int(self.intermediate_size)
            ),
            dropout=float(self.dropout),
        )


@dataclass(frozen=True)
class AnthropicComparison:
    prompt: str
    response_a: str
    response_b: str
    task: str
    selected: str
    preference_strength: str = "medium"
    source: str = "base"

    def __post_init__(self) -> None:
        if self.task not in {"helpfulness", "harmlessness"}:
            raise ValueError("task must be 'helpfulness' or 'harmlessness'")
        if self.selected not in {"a", "b"}:
            raise ValueError("selected must be 'a' or 'b'")

    def preferred_response(self) -> str:
        if self.task == "helpfulness":
            return self.response_a if self.selected == "a" else self.response_b
        return self.response_b if self.selected == "a" else self.response_a

    def dispreferred_response(self) -> str:
        if self.task == "helpfulness":
            return self.response_b if self.selected == "a" else self.response_a
        return self.response_a if self.selected == "a" else self.response_b


@dataclass(frozen=True)
class AnthropicFeedbackDataset:
    comparisons: tuple[AnthropicComparison, ...] = ()
    rejection_sampling_k: int = 16
    online_update_interval_days: int = 7

    def trainable_comparisons(self) -> tuple[AnthropicComparison, ...]:
        return tuple(
            comparison
            for comparison in self.comparisons
            if str(comparison.preference_strength).strip().lower() != "weakest"
        )

    def training_pairs(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            (
                comparison.preferred_response(),
                comparison.dispreferred_response(),
                comparison.task,
            )
            for comparison in self.trainable_comparisons()
        )

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for comparison in self.comparisons:
            counts[comparison.source] = counts.get(comparison.source, 0) + 1
        return counts


@dataclass(frozen=True)
class AnthropicRejectionSamplingChoice:
    objective: str
    selected_index: int
    num_candidates: int
    combined_scores: tuple[float, ...]


class AnthropicPreferenceModel(nn.Module):
    def __init__(self, config: AnthropicConfig) -> None:
        super().__init__()
        self.backbone = LLaMAModel(config.to_backbone_config())
        self.helpfulness_head = nn.Linear(int(config.hidden_size), 1, bias=False)
        self.harmlessness_head = nn.Linear(int(config.hidden_size), 1, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.backbone.forward_hidden(input_ids, attention_mask)
        pooled = _pool_last_hidden(hidden, attention_mask)
        return {
            "helpfulness": self.helpfulness_head(pooled).squeeze(-1),
            "harmlessness": self.harmlessness_head(pooled).squeeze(-1),
        }


class AnthropicModel(nn.Module):
    objective_names = ("helpfulness", "harmlessness")
    supports_online_rlhf = True

    def __init__(self, config: AnthropicConfig) -> None:
        super().__init__()
        self.config = config
        self.policy = LLaMAModel(config.to_backbone_config())
        self.reference_policy = deepcopy(self.policy)
        for param in self.reference_policy.parameters():
            param.requires_grad = False
        self.preference_model = AnthropicPreferenceModel(config)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.policy(input_ids, attention_mask)

    def preference_loss(
        self,
        *,
        chosen_input_ids: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor | None = None,
        rejected_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        chosen_scores = self.preference_model(chosen_input_ids, chosen_attention_mask)
        rejected_scores = self.preference_model(rejected_input_ids, rejected_attention_mask)
        helpfulness_loss = -F.logsigmoid(
            chosen_scores["helpfulness"] - rejected_scores["helpfulness"]
        ).mean()
        harmlessness_loss = -F.logsigmoid(
            chosen_scores["harmlessness"] - rejected_scores["harmlessness"]
        ).mean()
        return (
            float(self.config.helpfulness_mix) * helpfulness_loss
            + float(self.config.harmlessness_mix) * harmlessness_loss
        )

    def rejection_sample(
        self,
        *,
        candidate_input_ids: torch.Tensor,
        candidate_attention_mask: torch.Tensor | None = None,
        objective: str = "hh",
    ) -> AnthropicRejectionSamplingChoice:
        if candidate_input_ids.ndim != 2:
            raise ValueError("candidate_input_ids must have shape (num_candidates, seq_len)")
        resolved_objective = str(objective).strip().lower()
        if resolved_objective not in {"hh", "helpfulness", "harmlessness"}:
            raise ValueError("objective must be 'hh', 'helpfulness', or 'harmlessness'")

        scores = self.preference_model(candidate_input_ids, candidate_attention_mask)
        if resolved_objective == "hh":
            combined = (
                float(self.config.helpfulness_mix) * scores["helpfulness"]
                + float(self.config.harmlessness_mix) * scores["harmlessness"]
            )
        else:
            combined = scores[resolved_objective]

        selected_index = int(combined.argmax().item())
        return AnthropicRejectionSamplingChoice(
            objective=resolved_objective,
            selected_index=selected_index,
            num_candidates=int(candidate_input_ids.shape[0]),
            combined_scores=tuple(float(score) for score in combined.detach().cpu()),
        )


__all__ = [
    "AnthropicComparison",
    "AnthropicConfig",
    "AnthropicFeedbackDataset",
    "AnthropicModel",
    "AnthropicPreferenceModel",
    "AnthropicRejectionSamplingChoice",
]
