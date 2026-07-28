from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class TinyObservationEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        hidden = max(16, int(width) // 2)
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, int(width), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 5:
            raise ValueError(
                "Expected observations with shape (B, T, C, H, W), "
                f"got {tuple(observations.shape)}"
            )
        batch_size, steps = int(observations.shape[0]), int(observations.shape[1])
        encoded = self.net(observations.view(batch_size * steps, *observations.shape[2:]))
        return encoded.view(batch_size, steps, -1)


class TrajectoryEncoder(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(2, int(out_dim)),
            nn.ReLU(),
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.ndim != 3:
            raise ValueError(f"Expected trajectory shape (B, T, 2), got {tuple(trajectory.shape)}")
        return self.proj(trajectory.to(torch.float32))


class TextQuestionEncoder(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, out_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(int(vocab_size), int(out_dim), padding_idx=int(pad_id))
        self.proj = nn.Linear(int(out_dim), int(out_dim))

    def forward(self, question_ids: torch.Tensor, question_mask: torch.Tensor) -> torch.Tensor:
        if question_ids.ndim != 2:
            raise ValueError(f"Expected question_ids shape (B, L), got {tuple(question_ids.shape)}")
        emb = self.embed(question_ids.to(torch.long))
        mask = question_mask.to(torch.float32).unsqueeze(-1)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


@dataclass(frozen=True)
class EmbodiedQaConfig:
    vocab_size: int
    pad_id: int
    max_question_length: int
    trajectory_len: int
    num_answers: int = 4
    hidden_dim: int = 64
    vision_width: int = 32
    traj_width: int = 24
    text_width: int = 32


class CompactEmbodiedQaModel(nn.Module):
    def __init__(self, cfg: EmbodiedQaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.obs_encoder = TinyObservationEncoder(width=int(cfg.vision_width))
        self.traj_encoder = TrajectoryEncoder(out_dim=int(cfg.traj_width))
        self.text_encoder = TextQuestionEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            out_dim=int(cfg.text_width),
        )
        self.nav_rnn = nn.GRU(
            int(cfg.vision_width + cfg.traj_width),
            int(cfg.hidden_dim),
            batch_first=True,
        )
        self.fuse = nn.Sequential(
            nn.Linear(int(cfg.hidden_dim + cfg.text_width), int(cfg.hidden_dim)),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(int(cfg.hidden_dim), int(cfg.num_answers))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        trajectory = batch["trajectory"].to(torch.float32)
        observations = batch["observations"].to(torch.float32)
        question_ids = batch["question_ids"].to(torch.long)
        question_mask = batch["question_mask"].to(torch.float32)

        if int(trajectory.shape[1]) != int(self.cfg.trajectory_len):
            raise ValueError(
                "trajectory length does not match config: "
                f"{int(trajectory.shape[1])} != {int(self.cfg.trajectory_len)}"
            )
        if int(question_ids.shape[1]) != int(self.cfg.max_question_length):
            raise ValueError(
                "question length does not match config: "
                f"{int(question_ids.shape[1])} != {int(self.cfg.max_question_length)}"
            )

        obs_seq = self.obs_encoder(observations)
        traj_seq = self.traj_encoder(trajectory)
        nav_seq = torch.cat([obs_seq, traj_seq], dim=-1)
        _states, hidden = self.nav_rnn(nav_seq)
        nav_state = hidden[-1]

        question_state = self.text_encoder(question_ids, question_mask)
        fused_state = self.fuse(torch.cat([nav_state, question_state], dim=-1))
        logits = self.classifier(fused_state)
        return {"logits": logits, "fused_state": fused_state, "nav_state": nav_state}


def eqa_loss(logits: torch.Tensor, answer_id: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, answer_id.to(torch.long))


@torch.no_grad()
def eqa_accuracy(logits: torch.Tensor, answer_id: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    acc = (pred == answer_id.to(torch.long)).to(torch.float32).mean()
    return float(acc.item())


__all__ = [
    "EmbodiedQaConfig",
    "TextQuestionEncoder",
    "TinyObservationEncoder",
    "CompactEmbodiedQaModel",
    "TrajectoryEncoder",
    "eqa_accuracy",
    "eqa_loss",
]
