from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .llama import LLaMAConfig, LLaMAModel


def _gather_token_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=token_ids.to(torch.long).unsqueeze(-1)).squeeze(-1)


def _token_kl_divergence(policy_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
    reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
    policy_probs = policy_log_probs.exp()
    return (policy_probs * (policy_log_probs - reference_log_probs)).sum(dim=-1)


def _pool_last_hidden(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if attention_mask is None:
        return hidden[:, -1]
    last_index = attention_mask.to(torch.long).sum(dim=1).clamp_min(1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last_index]


@dataclass(frozen=True)
class InstructGPTConfig:
    vocab_size: int
    max_seq_len: int
    hidden_size: int = 512
    num_heads: int = 8
    num_layers: int = 6
    intermediate_size: int | None = None
    dropout: float = 0.0
    kl_coeff: float = 0.02
    pretraining_mix_coeff: float = 27.8
    clip_range: float = 0.2

    def to_backbone_config(self) -> LLaMAConfig:
        return LLaMAConfig(
            vocab_size=int(self.vocab_size),
            max_seq_len=int(self.max_seq_len),
            dim=int(self.hidden_size),
            n_heads=int(self.num_heads),
            n_layers=int(self.num_layers),
            intermediate_size=self.intermediate_size,
            dropout=float(self.dropout),
        )


class InstructGPTRewardModel(nn.Module):
    def __init__(self, config: InstructGPTConfig) -> None:
        super().__init__()
        self.backbone = LLaMAModel(config.to_backbone_config())
        self.reward_head = nn.Linear(int(config.hidden_size), 1, bias=False)
        self.reward_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.backbone.forward_hidden(input_ids, attention_mask)
        pooled = _pool_last_hidden(hidden, attention_mask)
        return self.reward_head(pooled).squeeze(-1) + self.reward_bias

    @torch.no_grad()
    def set_reward_bias_from_demonstrations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        hidden = self.backbone.forward_hidden(input_ids, attention_mask)
        pooled = _pool_last_hidden(hidden, attention_mask)
        base_rewards = self.reward_head(pooled).squeeze(-1)
        self.reward_bias.copy_(-base_rewards.mean().reshape_as(self.reward_bias))


class InstructGPTValueModel(InstructGPTRewardModel):
    pass


class PPOPTXObjective(nn.Module):
    def __init__(
        self,
        *,
        kl_coeff: float,
        pretraining_mix_coeff: float,
        clip_range: float = 0.2,
    ) -> None:
        super().__init__()
        self.kl_coeff = float(kl_coeff)
        self.pretraining_mix_coeff = float(pretraining_mix_coeff)
        self.clip_range = float(clip_range)

    def forward(
        self,
        *,
        policy_logits: torch.Tensor,
        old_policy_logits: torch.Tensor | None = None,
        reference_logits: torch.Tensor,
        sampled_ids: torch.Tensor,
        advantages: torch.Tensor,
        pretraining_logits: torch.Tensor | None = None,
        pretraining_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        policy_logprobs = _gather_token_logprobs(policy_logits, sampled_ids)
        old_policy_logprobs = _gather_token_logprobs(
            reference_logits if old_policy_logits is None else old_policy_logits,
            sampled_ids,
        )
        advantages = advantages.to(dtype=policy_logits.dtype, device=policy_logits.device)

        ratios = torch.exp(policy_logprobs - old_policy_logprobs.detach())
        clipped_ratios = ratios.clamp(1.0 - self.clip_range, 1.0 + self.clip_range)
        surrogate = -torch.minimum(ratios * advantages, clipped_ratios * advantages).mean()
        kl_penalty = self.kl_coeff * _token_kl_divergence(policy_logits, reference_logits).mean()
        loss = surrogate + kl_penalty

        if pretraining_logits is not None and pretraining_labels is not None:
            ptx_loss = F.cross_entropy(
                pretraining_logits.reshape(-1, pretraining_logits.shape[-1]),
                pretraining_labels.to(torch.long).reshape(-1),
            )
            loss = loss + (self.pretraining_mix_coeff * ptx_loss)

        return loss


class InstructGPTModel(nn.Module):
    stage_order = ("sft", "reward_model", "ppo")

    def __init__(self, config: InstructGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.policy = LLaMAModel(config.to_backbone_config())
        self.reference_policy = deepcopy(self.policy)
        for param in self.reference_policy.parameters():
            param.requires_grad = False
        self.reward_model = InstructGPTRewardModel(config)
        self.value_model = InstructGPTValueModel(config)
        self.value_model.load_state_dict(self.reward_model.state_dict())
        self.objective = PPOPTXObjective(
            kl_coeff=float(config.kl_coeff),
            pretraining_mix_coeff=float(config.pretraining_mix_coeff),
            clip_range=float(config.clip_range),
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.policy(input_ids, attention_mask)

    def sft_loss(
        self,
        *,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.to(torch.long).reshape(-1),
        )

    def reward_loss(
        self,
        *,
        chosen_input_ids: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor | None = None,
        rejected_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        chosen_rewards = self.reward_model(chosen_input_ids, chosen_attention_mask)
        rejected_rewards = self.reward_model(rejected_input_ids, rejected_attention_mask)
        return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    def reward_loss_from_rankings(
        self,
        *,
        completion_input_ids: torch.Tensor,
        rankings: torch.Tensor,
        completion_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if completion_input_ids.ndim != 3:
            raise ValueError("completion_input_ids must have shape (batch, completions, seq_len)")
        if rankings.shape != completion_input_ids.shape[:2]:
            raise ValueError(
                "rankings must have shape matching the batch and completions dimensions of completion_input_ids"
            )
        if completion_input_ids.shape[1] < 2:
            raise ValueError("ranked reward loss requires at least two completions per prompt")

        batch_size, num_completions, seq_len = completion_input_ids.shape
        flat_input_ids = completion_input_ids.reshape(batch_size * num_completions, seq_len)
        flat_attention_mask = None
        if completion_attention_mask is not None:
            if completion_attention_mask.shape != completion_input_ids.shape:
                raise ValueError("completion_attention_mask must match completion_input_ids")
            flat_attention_mask = completion_attention_mask.reshape(batch_size * num_completions, seq_len)

        rewards = self.reward_model(flat_input_ids, flat_attention_mask).view(batch_size, num_completions)

        pairwise_losses: list[torch.Tensor] = []
        for left in range(num_completions):
            for right in range(left + 1, num_completions):
                left_is_better = rankings[:, left] < rankings[:, right]
                better_rewards = torch.where(left_is_better, rewards[:, left], rewards[:, right])
                worse_rewards = torch.where(left_is_better, rewards[:, right], rewards[:, left])
                pairwise_losses.append(-F.logsigmoid(better_rewards - worse_rewards))
        return torch.stack(pairwise_losses, dim=0).mean()


__all__ = [
    "InstructGPTConfig",
    "InstructGPTModel",
    "InstructGPTRewardModel",
    "InstructGPTValueModel",
    "PPOPTXObjective",
]
