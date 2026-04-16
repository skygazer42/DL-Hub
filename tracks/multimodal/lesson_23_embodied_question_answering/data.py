from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

_DIRECTIONS: tuple[str, ...] = ("left", "right", "up", "down")
_DIRECTION_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(_DIRECTIONS)}
_DIR_STEP: dict[str, tuple[float, float]] = {
    "left": (0.0, -1.0),
    "right": (0.0, 1.0),
    "up": (-1.0, 0.0),
    "down": (1.0, 0.0),
}


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_token: str = "<pad>"

    @property
    def pad_id(self) -> int:
        return int(self.token_to_id[self.pad_token])

    @property
    def size(self) -> int:
        return int(len(self.id_to_token))

    def encode(self, tokens: list[str], *, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if len(tokens) > int(max_length):
            raise ValueError(f"Question exceeds max_length={int(max_length)}")
        ids = [int(self.token_to_id[token]) for token in tokens]
        pad = int(max_length) - len(ids)
        ids.extend([self.pad_id] * pad)
        mask = [1.0] * (int(max_length) - pad) + [0.0] * pad
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "pad_token": str(self.pad_token),
            "pad_id": int(self.pad_id),
            "token_to_id": {k: int(v) for k, v in self.token_to_id.items()},
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 512
    batch_size: int = 16
    trajectory_len: int = 6
    image_size: int = 20
    max_question_length: int = 12
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "where",
        "is",
        "the",
        "goal",
        "from",
        "final",
        "position",
        "left",
        "right",
        "up",
        "down",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _validate_config(cfg: DataConfig) -> None:
    if int(cfg.num_samples) <= 0:
        raise ValueError("num_samples must be positive")
    if int(cfg.trajectory_len) < 4:
        raise ValueError("trajectory_len must be >= 4")
    if int(cfg.image_size) < 16:
        raise ValueError("image_size must be >= 16")
    if int(cfg.max_question_length) < 8:
        raise ValueError("max_question_length must be >= 8")


def _render_frame(
    *,
    image_size: int,
    agent_y: float,
    agent_x: float,
    goal_y: float,
    goal_x: float,
) -> torch.Tensor:
    image = torch.full((3, int(image_size), int(image_size)), 0.04, dtype=torch.float32)
    ys = torch.arange(int(image_size), dtype=torch.float32).view(-1, 1).expand(int(image_size), int(image_size))
    xs = torch.arange(int(image_size), dtype=torch.float32).view(1, -1).expand(int(image_size), int(image_size))

    agent_mask = (ys - float(agent_y)).abs() <= 1.5
    agent_mask = agent_mask & ((xs - float(agent_x)).abs() <= 1.5)
    goal_mask = (ys - float(goal_y)).abs() <= 1.5
    goal_mask = goal_mask & ((xs - float(goal_x)).abs() <= 1.5)

    agent_color = torch.tensor([0.2, 0.8, 1.0], dtype=torch.float32).view(3, 1, 1)
    goal_color = torch.tensor([1.0, 0.8, 0.2], dtype=torch.float32).view(3, 1, 1)
    image = torch.where(agent_mask.unsqueeze(0), agent_color, image)
    image = torch.where(goal_mask.unsqueeze(0), goal_color, image)

    image[:, 0, :] = 0.08
    image[:, -1, :] = 0.08
    image[:, :, 0] = 0.08
    image[:, :, -1] = 0.08
    return image.clamp(0.0, 1.0)


class ToyEmbodiedQaDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        _validate_config(cfg)
        self.cfg = cfg
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        generator = torch.Generator().manual_seed(int(self.cfg.seed) + int(idx))
        direction = _DIRECTIONS[int(idx) % len(_DIRECTIONS)]
        answer_id = int(_DIRECTION_TO_ID[direction])
        dy, dx = _DIR_STEP[direction]
        img = int(self.cfg.image_size)
        traj_len = int(self.cfg.trajectory_len)

        margin = 4.0
        max_step = 1.2 * float(traj_len - 1)
        start_y = float(torch.empty(1).uniform_(margin + max_step, img - margin - max_step, generator=generator).item())
        start_x = float(torch.empty(1).uniform_(margin + max_step, img - margin - max_step, generator=generator).item())

        trajectory: list[torch.Tensor] = []
        observations: list[torch.Tensor] = []
        positions: list[tuple[float, float]] = []
        for step in range(traj_len):
            y = start_y + 1.2 * float(step) * dy
            x = start_x + 1.2 * float(step) * dx
            y = float(max(margin, min(float(img - margin), y)))
            x = float(max(margin, min(float(img - margin), x)))
            positions.append((y, x))

        final_y, final_x = positions[-1]
        goal_y = final_y + 3.0 * dy
        goal_x = final_x + 3.0 * dx
        goal_y = float(max(2.0, min(float(img - 3), goal_y)))
        goal_x = float(max(2.0, min(float(img - 3), goal_x)))

        for y, x in positions:
            trajectory.append(torch.tensor([y / float(img - 1), x / float(img - 1)], dtype=torch.float32))
            observations.append(
                _render_frame(
                    image_size=img,
                    agent_y=y,
                    agent_x=x,
                    goal_y=goal_y,
                    goal_x=goal_x,
                )
            )

        question_tokens = ["where", "is", "the", "goal", "from", "final", "position"]
        question_ids, question_mask = self.vocab.encode(
            question_tokens,
            max_length=int(self.cfg.max_question_length),
        )

        return {
            "trajectory": torch.stack(trajectory, dim=0),
            "observations": torch.stack(observations, dim=0),
            "question_ids": question_ids,
            "question_mask": question_mask,
            "answer_id": torch.tensor(answer_id, dtype=torch.long),
            "target_step": torch.tensor(traj_len - 1, dtype=torch.long),
            "question_text": " ".join(question_tokens),
            "answer_text": direction,
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Vocab]:
    vocab = _build_vocab()
    dataset = ToyEmbodiedQaDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "trajectory": torch.stack([sample["trajectory"] for sample in batch], dim=0),
            "observations": torch.stack([sample["observations"] for sample in batch], dim=0),
            "question_ids": torch.stack([sample["question_ids"] for sample in batch], dim=0),
            "question_mask": torch.stack([sample["question_mask"] for sample in batch], dim=0),
            "answer_id": torch.stack([sample["answer_id"] for sample in batch], dim=0),
            "target_step": torch.stack([sample["target_step"] for sample in batch], dim=0),
            "question_text": [str(sample["question_text"]) for sample in batch],
            "answer_text": [str(sample["answer_text"]) for sample in batch],
        }

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    return train_loader, val_loader, vocab


__all__ = ["DataConfig", "ToyEmbodiedQaDataset", "Vocab", "get_dataloaders"]
