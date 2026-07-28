from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

_SHIRT_COLORS: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.2, 0.2),
    "green": (0.2, 0.85, 0.25),
    "blue": (0.25, 0.4, 1.0),
    "yellow": (1.0, 0.9, 0.2),
}
_PANTS_COLORS: dict[str, tuple[float, float, float]] = {
    "black": (0.15, 0.15, 0.15),
    "gray": (0.5, 0.5, 0.5),
    "white": (0.88, 0.88, 0.88),
}
_ACCESSORIES: tuple[str, ...] = ("backpack", "hat", "none")


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
            raise ValueError(
                f"Too many tokens for max_length={int(max_length)}: got {len(tokens)} tokens."
            )
        input_ids = [int(self.token_to_id[token]) for token in tokens]
        pad_count = int(max_length) - len(input_ids)
        input_ids.extend([self.pad_id] * pad_count)
        attention_mask = [1.0] * (int(max_length) - pad_count) + [0.0] * pad_count
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.float32),
        )

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
    image_size: int = 24
    max_text_length: int = 10
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "person",
        *_SHIRT_COLORS.keys(),
        *_PANTS_COLORS.keys(),
        *_ACCESSORIES,
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _all_profiles() -> list[tuple[str, str, str]]:
    return list(product(_SHIRT_COLORS.keys(), _PANTS_COLORS.keys(), _ACCESSORIES))


def _validate_config(cfg: DataConfig) -> None:
    if int(cfg.num_samples) <= 0:
        raise ValueError("num_samples must be positive")
    if int(cfg.image_size) < 20:
        raise ValueError("image_size must be >= 20")
    if int(cfg.max_text_length) < 4:
        raise ValueError("max_text_length must be >= 4")


def _make_profile_list(cfg: DataConfig) -> list[tuple[str, str, str]]:
    profiles = _all_profiles()
    gen = torch.Generator().manual_seed(int(cfg.seed))
    order = torch.randperm(len(profiles), generator=gen).tolist()
    shuffled = [profiles[idx] for idx in order]
    out: list[tuple[str, str, str]] = []
    while len(out) < int(cfg.num_samples):
        out.extend(shuffled)
    return out[: int(cfg.num_samples)]


def _render_person(
    *,
    shirt_name: str,
    pants_name: str,
    accessory_name: str,
    image_size: int,
) -> torch.Tensor:
    size = int(image_size)
    image = torch.full((3, size, size), 0.04, dtype=torch.float32)
    shirt_color = torch.tensor(_SHIRT_COLORS[shirt_name], dtype=torch.float32).view(3, 1, 1)
    pants_color = torch.tensor(_PANTS_COLORS[pants_name], dtype=torch.float32).view(3, 1, 1)

    ys = torch.arange(size, dtype=torch.float32).view(-1, 1).expand(size, size)
    xs = torch.arange(size, dtype=torch.float32).view(1, -1).expand(size, size)
    center = 0.5 * (size - 1)

    head_mask = ((ys - 0.22 * size).pow(2) + (xs - center).pow(2)) <= float((0.11 * size) ** 2)
    torso_mask = (
        (ys >= 0.32 * size)
        & (ys <= 0.63 * size)
        & ((xs - center).abs() <= 0.18 * size)
    )
    leg_left = (ys >= 0.64 * size) & (ys <= 0.92 * size) & (xs >= 0.36 * size) & (xs <= 0.48 * size)
    leg_right = (ys >= 0.64 * size) & (ys <= 0.92 * size) & (xs >= 0.52 * size) & (xs <= 0.64 * size)
    pants_mask = leg_left | leg_right

    image = torch.where(torso_mask.unsqueeze(0), shirt_color.expand_as(image), image)
    image = torch.where(pants_mask.unsqueeze(0), pants_color.expand_as(image), image)
    skin_color = torch.tensor((0.95, 0.8, 0.7), dtype=torch.float32).view(3, 1, 1)
    image = torch.where(head_mask.unsqueeze(0), skin_color.expand_as(image), image)

    if accessory_name == "backpack":
        bag_mask = (ys >= 0.4 * size) & (ys <= 0.66 * size) & (xs >= 0.66 * size) & (xs <= 0.78 * size)
        bag_color = torch.tensor((0.45, 0.2, 0.12), dtype=torch.float32).view(3, 1, 1)
        image = torch.where(bag_mask.unsqueeze(0), bag_color.expand_as(image), image)
    elif accessory_name == "hat":
        hat_mask = (ys >= 0.08 * size) & (ys <= 0.16 * size) & (xs >= 0.36 * size) & (xs <= 0.64 * size)
        hat_color = torch.tensor((0.05, 0.05, 0.05), dtype=torch.float32).view(3, 1, 1)
        image = torch.where(hat_mask.unsqueeze(0), hat_color.expand_as(image), image)

    image[:, 0, :] = 0.08
    image[:, -1, :] = 0.08
    image[:, :, 0] = 0.08
    image[:, :, -1] = 0.08
    return image.clamp(0.0, 1.0)


class SyntheticPersonSearchDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        _validate_config(cfg)
        self.cfg = cfg
        self.vocab = vocab
        self.profiles = _make_profile_list(cfg)

    def __len__(self) -> int:
        return int(len(self.profiles))

    def __getitem__(self, idx: int) -> dict[str, object]:
        shirt_name, pants_name, accessory_name = self.profiles[int(idx)]
        tokens = ["person", shirt_name, pants_name, accessory_name]
        input_ids, attention_mask = self.vocab.encode(tokens, max_length=int(self.cfg.max_text_length))
        image = _render_person(
            shirt_name=shirt_name,
            pants_name=pants_name,
            accessory_name=accessory_name,
            image_size=int(self.cfg.image_size),
        )
        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "person_id": torch.tensor(int(idx), dtype=torch.long),
            "query_text": " ".join(tokens),
            "attribute_tokens": tokens,
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Vocab]:
    vocab = _build_vocab()
    dataset = SyntheticPersonSearchDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "image": torch.stack([sample["image"] for sample in batch], dim=0),
            "input_ids": torch.stack([sample["input_ids"] for sample in batch], dim=0),
            "attention_mask": torch.stack([sample["attention_mask"] for sample in batch], dim=0),
            "person_id": torch.stack([sample["person_id"] for sample in batch], dim=0),
            "query_text": [str(sample["query_text"]) for sample in batch],
            "attribute_tokens": [list(sample["attribute_tokens"]) for sample in batch],
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


__all__ = ["DataConfig", "SyntheticPersonSearchDataset", "Vocab", "get_dataloaders"]
