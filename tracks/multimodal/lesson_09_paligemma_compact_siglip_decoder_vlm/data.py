from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

_COLORS: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.15, 0.15),
    "green": (0.15, 0.95, 0.2),
    "blue": (0.2, 0.35, 1.0),
    "yellow": (1.0, 0.9, 0.15),
}
_SHAPES: tuple[str, ...] = ("square", "circle", "cross")
_SIZES: tuple[str, ...] = ("small", "large")
_LOCATIONS: tuple[tuple[str, str], ...] = (
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
)
_TASK_TYPES: tuple[str, ...] = ("caption", "color", "shape", "locate", "yes_no")


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    sep_token: str = "<sep>"

    @property
    def pad_id(self) -> int:
        return int(self.token_to_id[self.pad_token])

    @property
    def bos_id(self) -> int:
        return int(self.token_to_id[self.bos_token])

    @property
    def eos_id(self) -> int:
        return int(self.token_to_id[self.eos_token])

    @property
    def sep_id(self) -> int:
        return int(self.token_to_id[self.sep_token])

    @property
    def size(self) -> int:
        return int(len(self.id_to_token))

    def encode_prompt(self, tokens: list[str], *, max_length: int) -> torch.Tensor:
        seq = [self.bos_id, *[int(self.token_to_id[token]) for token in tokens], self.sep_id]
        if len(seq) > int(max_length):
            raise ValueError(f"Prompt exceeds max_length={int(max_length)}.")
        pad_count = int(max_length) - len(seq)
        seq.extend([self.pad_id] * pad_count)
        return torch.tensor(seq, dtype=torch.long)

    def encode_example(
        self, prompt_tokens: list[str], answer_tokens: list[str], *, max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_ids = [self.bos_id, *[int(self.token_to_id[token]) for token in prompt_tokens], self.sep_id]
        answer_ids = [int(self.token_to_id[token]) for token in answer_tokens]
        full_ids = [*prompt_ids, *answer_ids]
        if len(full_ids) > int(max_length):
            raise ValueError(f"Example exceeds max_length={int(max_length)}.")

        labels = [-100] * (len(prompt_ids) - 1) + [*answer_ids, self.eos_id]
        pad_count = int(max_length) - len(full_ids)
        full_ids.extend([self.pad_id] * pad_count)
        labels.extend([-100] * pad_count)
        mask = [1.0] * (int(max_length) - pad_count) + [0.0] * pad_count

        return (
            torch.tensor(full_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32),
        )

    def decode_ids(self, ids: list[int]) -> list[str]:
        tokens: list[str] = []
        for idx in ids:
            token = self.id_to_token[int(idx)]
            if token in {self.pad_token, self.bos_token, self.sep_token}:
                continue
            if token == self.eos_token:
                break
            tokens.append(token)
        return tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "pad_id": int(self.pad_id),
            "bos_id": int(self.bos_id),
            "eos_id": int(self.eos_id),
            "sep_id": int(self.sep_id),
            "token_to_id": {k: int(v) for k, v in self.token_to_id.items()},
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 512
    batch_size: int = 32
    image_size: int = 16
    max_text_length: int = 16
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<sep>",
        "caption",
        "answer",
        "locate",
        "the",
        "image",
        "object",
        "color",
        "shape",
        "is",
        "red",
        "green",
        "blue",
        "yellow",
        "square",
        "circle",
        "cross",
        "small",
        "large",
        "top",
        "bottom",
        "left",
        "right",
        "yes",
        "no",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _all_concepts() -> list[tuple[str, str, str, tuple[str, str]]]:
    return list(product(_COLORS.keys(), _SHAPES, _SIZES, _LOCATIONS))


def _render_shape(
    *,
    image_size: int,
    color_name: str,
    shape_name: str,
    size_name: str,
    location: tuple[str, str],
) -> torch.Tensor:
    image = torch.zeros(3, int(image_size), int(image_size), dtype=torch.float32)
    color = torch.tensor(_COLORS[color_name], dtype=torch.float32).view(3, 1, 1)

    vertical, horizontal = location
    center_y = int(round((0.28 if vertical == "top" else 0.72) * (int(image_size) - 1)))
    center_x = int(round((0.28 if horizontal == "left" else 0.72) * (int(image_size) - 1)))
    radius = max(2, int(round((0.16 if size_name == "small" else 0.26) * int(image_size))))

    ys = torch.arange(int(image_size), dtype=torch.float32).view(-1, 1)
    xs = torch.arange(int(image_size), dtype=torch.float32).view(1, -1)
    yy = ys.expand(int(image_size), int(image_size))
    xx = xs.expand(int(image_size), int(image_size))

    if shape_name == "square":
        mask = (yy - center_y).abs() <= radius
        mask = mask & ((xx - center_x).abs() <= radius)
    elif shape_name == "circle":
        mask = (yy - center_y).pow(2) + (xx - center_x).pow(2) <= float(radius * radius)
    elif shape_name == "cross":
        thickness = max(1, radius // 2)
        vertical_mask = (xx - center_x).abs() <= thickness
        horizontal_mask = (yy - center_y).abs() <= thickness
        arm = (yy - center_y).abs() <= radius
        arm = arm & ((xx - center_x).abs() <= radius)
        mask = (vertical_mask & arm) | (horizontal_mask & arm)
    else:
        raise ValueError(f"Unsupported shape: {shape_name}")

    image[:] = 0.03
    image = torch.where(mask.unsqueeze(0), color.expand_as(image), image)
    image[:, 0, :] = 0.08
    image[:, -1, :] = 0.08
    image[:, :, 0] = 0.08
    image[:, :, -1] = 0.08
    return image.clamp(0.0, 1.0)


def _make_prompt_answer(
    concept: tuple[str, str, str, tuple[str, str]],
    *,
    idx: int,
) -> tuple[str, list[str], list[str]]:
    color_name, shape_name, _size_name, location = concept
    task_type = _TASK_TYPES[int(idx) % len(_TASK_TYPES)]

    if task_type == "caption":
        return task_type, ["caption", "the", "image"], [color_name, shape_name, location[0], location[1]]
    if task_type == "color":
        return task_type, ["answer", "color"], [color_name]
    if task_type == "shape":
        return task_type, ["answer", "shape"], [shape_name]
    if task_type == "locate":
        return task_type, ["locate", "the", "object"], [location[0], location[1]]

    truth = ((int(idx) // len(_TASK_TYPES)) % 2) == 0
    target_color = color_name if truth else next(name for name in _COLORS if name != color_name)
    return "yes_no", ["is", "the", "object", target_color], ["yes" if truth else "no"]


class SyntheticPaliGemmaDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        self.cfg = cfg
        self.vocab = vocab
        concepts = _all_concepts()
        if int(cfg.num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        if int(cfg.image_size) < 12:
            raise ValueError("image_size must be >= 12")
        if int(cfg.max_text_length) < 12:
            raise ValueError("max_text_length must be >= 12")

        generator = torch.Generator().manual_seed(int(cfg.seed))
        order = torch.randperm(len(concepts), generator=generator).tolist()
        shuffled = [concepts[idx] for idx in order]
        repeated: list[tuple[str, str, str, tuple[str, str]]] = []
        while len(repeated) < int(cfg.num_samples):
            repeated.extend(shuffled)
        self.concepts = repeated[: int(cfg.num_samples)]
        self.records = [self._build_record(idx, concept) for idx, concept in enumerate(self.concepts)]

    def _build_record(
        self, idx: int, concept: tuple[str, str, str, tuple[str, str]]
    ) -> dict[str, object]:
        color_name, shape_name, size_name, location = concept
        task_type, prompt_tokens, answer_tokens = _make_prompt_answer(concept, idx=idx)
        prompt_ids = self.vocab.encode_prompt(prompt_tokens, max_length=int(self.cfg.max_text_length))
        input_ids, labels, attention_mask = self.vocab.encode_example(
            prompt_tokens,
            answer_tokens,
            max_length=int(self.cfg.max_text_length),
        )
        image = _render_shape(
            image_size=int(self.cfg.image_size),
            color_name=color_name,
            shape_name=shape_name,
            size_name=size_name,
            location=location,
        )
        return {
            "image": image,
            "prompt_ids": prompt_ids,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "task_type": task_type,
            "prompt_text": " ".join(prompt_tokens),
            "answer_text": " ".join(answer_tokens),
        }

    def __len__(self) -> int:
        return int(len(self.records))

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.records[int(idx)]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Vocab]:
    vocab = _build_vocab()
    dataset = SyntheticPaliGemmaDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "image": torch.stack([sample["image"] for sample in batch], dim=0),
            "prompt_ids": torch.stack([sample["prompt_ids"] for sample in batch], dim=0),
            "input_ids": torch.stack([sample["input_ids"] for sample in batch], dim=0),
            "labels": torch.stack([sample["labels"] for sample in batch], dim=0),
            "attention_mask": torch.stack([sample["attention_mask"] for sample in batch], dim=0),
            "task_type": [str(sample["task_type"]) for sample in batch],
            "prompt_text": [str(sample["prompt_text"]) for sample in batch],
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


__all__ = ["DataConfig", "SyntheticPaliGemmaDataset", "Vocab", "get_dataloaders"]
