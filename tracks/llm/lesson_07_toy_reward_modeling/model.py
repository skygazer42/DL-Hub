from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    embed_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.1


class ToyRewardModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(
            num_embeddings=int(cfg.vocab_size),
            embedding_dim=int(cfg.embed_dim),
            padding_idx=int(cfg.pad_id),
        )
        self.encoder = nn.GRU(
            input_size=int(cfg.embed_dim),
            hidden_size=int(cfg.hidden_dim),
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.reward_head = nn.Linear(int(cfg.hidden_dim), 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        states, _ = self.encoder(embedded)
        lengths = attention_mask.sum(dim=1).to(torch.long).clamp(min=1) - 1
        batch_idx = torch.arange(states.shape[0], device=states.device)
        pooled = states[batch_idx, lengths]
        rewards = self.reward_head(self.dropout(pooled)).squeeze(-1)
        return rewards

    def preference_loss(self, *, chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
        return F.softplus(-(chosen_rewards - rejected_rewards)).mean()


def preference_accuracy(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> float:
    return float((chosen_rewards > rejected_rewards).to(torch.float32).mean().item())


__all__ = ["ModelConfig", "ToyRewardModel", "preference_accuracy"]
