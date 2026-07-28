from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

_ACTIONS: tuple[str, ...] = ("north", "south", "east", "west")
_ACTION_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(_ACTIONS)}
_STEP_DELTA: dict[str, tuple[int, int]] = {
    "north": (-1, 0),
    "south": (1, 0),
    "east": (0, 1),
    "west": (0, -1),
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
            raise ValueError(f"Instruction exceeds max_length={int(max_length)}")
        ids = [int(self.token_to_id[token]) for token in tokens]
        pad = int(max_length) - len(ids)
        ids.extend([self.pad_id] * pad)
        mask = [1.0] * (int(max_length) - pad) + [0.0] * pad
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 512
    batch_size: int = 16
    grid_size: int = 7
    max_instruction_length: int = 12
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "move",
        "one",
        "step",
        "toward",
        "the",
        "goal",
        "north",
        "south",
        "east",
        "west",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _validate_config(cfg: DataConfig) -> None:
    if int(cfg.num_samples) <= 0:
        raise ValueError("num_samples must be positive")
    if int(cfg.grid_size) < 5:
        raise ValueError("grid_size must be >= 5")
    if int(cfg.max_instruction_length) < 7:
        raise ValueError("max_instruction_length must be >= 7")


def _sample_positions(grid_size: int, generator: torch.Generator) -> tuple[tuple[int, int], tuple[int, int]]:
    while True:
        agent = (
            int(torch.randint(0, int(grid_size), (1,), generator=generator).item()),
            int(torch.randint(0, int(grid_size), (1,), generator=generator).item()),
        )
        goal = (
            int(torch.randint(0, int(grid_size), (1,), generator=generator).item()),
            int(torch.randint(0, int(grid_size), (1,), generator=generator).item()),
        )
        if agent != goal:
            return agent, goal


def _target_action(agent_pos: tuple[int, int], goal_pos: tuple[int, int]) -> str:
    ay, ax = agent_pos
    gy, gx = goal_pos
    if gy < ay:
        return "north"
    if gy > ay:
        return "south"
    if gx > ax:
        return "east"
    return "west"


def _render_observation(
    *,
    grid_size: int,
    agent_pos: tuple[int, int],
    goal_pos: tuple[int, int],
) -> torch.Tensor:
    image = torch.full((3, int(grid_size), int(grid_size)), 0.03, dtype=torch.float32)
    for row in range(int(grid_size)):
        for col in range(int(grid_size)):
            if row == 0 or row == int(grid_size) - 1 or col == 0 or col == int(grid_size) - 1:
                image[:, row, col] = 0.08
    ay, ax = agent_pos
    gy, gx = goal_pos
    image[:, ay, ax] = torch.tensor([0.15, 0.85, 1.0], dtype=torch.float32)
    image[:, gy, gx] = torch.tensor([1.0, 0.8, 0.2], dtype=torch.float32)
    return image.clamp(0.0, 1.0)


class SyntheticVisionLanguageNavigationDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        _validate_config(cfg)
        self.cfg = cfg
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        generator = torch.Generator().manual_seed(int(self.cfg.seed) + int(idx))
        agent_pos, goal_pos = _sample_positions(int(self.cfg.grid_size), generator)
        action = _target_action(agent_pos, goal_pos)
        action_id = int(_ACTION_TO_ID[action])

        instruction_tokens = ["move", "one", "step", "toward", "the", "goal", action]
        instruction_ids, instruction_mask = self.vocab.encode(
            instruction_tokens,
            max_length=int(self.cfg.max_instruction_length),
        )

        return {
            "observation": _render_observation(
                grid_size=int(self.cfg.grid_size),
                agent_pos=agent_pos,
                goal_pos=goal_pos,
            ),
            "instruction_ids": instruction_ids,
            "instruction_mask": instruction_mask,
            "actions": torch.tensor(action_id, dtype=torch.long),
            "agent_pos": torch.tensor(agent_pos, dtype=torch.long),
            "goal_pos": torch.tensor(goal_pos, dtype=torch.long),
            "instruction_text": " ".join(instruction_tokens),
            "action_text": action,
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Vocab]:
    vocab = _build_vocab()
    dataset = SyntheticVisionLanguageNavigationDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "observation": torch.stack([sample["observation"] for sample in batch], dim=0),
            "instruction_ids": torch.stack([sample["instruction_ids"] for sample in batch], dim=0),
            "instruction_mask": torch.stack([sample["instruction_mask"] for sample in batch], dim=0),
            "actions": torch.stack([sample["actions"] for sample in batch], dim=0),
            "agent_pos": torch.stack([sample["agent_pos"] for sample in batch], dim=0),
            "goal_pos": torch.stack([sample["goal_pos"] for sample in batch], dim=0),
            "instruction_text": [str(sample["instruction_text"]) for sample in batch],
            "action_text": [str(sample["action_text"]) for sample in batch],
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


__all__ = ["DataConfig", "SyntheticVisionLanguageNavigationDataset", "Vocab", "get_dataloaders"]
