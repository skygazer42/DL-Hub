from __future__ import annotations

from dataclasses import dataclass

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
_DIRECTIONS: tuple[str, ...] = ("left", "right", "up", "down")
_TASK_TYPES: tuple[str, ...] = ("color", "shape", "left", "right", "up", "down")


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
    seq_len: int = 4
    image_size: int = 20
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
        "what",
        "color",
        "shape",
        "is",
        "it",
        "moving",
        "the",
        "object",
        "left",
        "right",
        "up",
        "down",
        "red",
        "green",
        "blue",
        "yellow",
        "square",
        "circle",
        "cross",
        "yes",
        "no",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _shape_mask(
    *,
    height: int,
    width: int,
    shape_name: str,
    center_x: float,
    center_y: float,
    radius: float,
) -> torch.Tensor:
    ys = torch.arange(height, dtype=torch.float32).view(-1, 1).expand(height, width)
    xs = torch.arange(width, dtype=torch.float32).view(1, -1).expand(height, width)

    if shape_name == "square":
        mask = (xs - float(center_x)).abs() <= float(radius)
        return mask & ((ys - float(center_y)).abs() <= float(radius))

    if shape_name == "circle":
        return (xs - float(center_x)).pow(2) + (ys - float(center_y)).pow(2) <= float(radius * radius)

    if shape_name == "cross":
        thickness = max(1.0, float(radius) / 2.0)
        vertical = (xs - float(center_x)).abs() <= thickness
        vertical = vertical & ((ys - float(center_y)).abs() <= float(radius))
        horizontal = (ys - float(center_y)).abs() <= thickness
        horizontal = horizontal & ((xs - float(center_x)).abs() <= float(radius))
        return vertical | horizontal

    raise ValueError(f"Unsupported shape: {shape_name}")


def _render_frame(
    *,
    image_size: int,
    color_name: str,
    shape_name: str,
    center_x: float,
    center_y: float,
    radius: float,
    generator: torch.Generator,
) -> torch.Tensor:
    image = 0.02 * torch.rand((3, int(image_size), int(image_size)), generator=generator, dtype=torch.float32)
    image = image + 0.02
    mask = _shape_mask(
        height=int(image_size),
        width=int(image_size),
        shape_name=shape_name,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
    )
    color = torch.tensor(_COLORS[color_name], dtype=torch.float32).view(3, 1, 1)
    image[:] = torch.where(mask.unsqueeze(0), color.expand_as(image), image)
    image[:, 0, :] = 0.08
    image[:, -1, :] = 0.08
    image[:, :, 0] = 0.08
    image[:, :, -1] = 0.08
    return image.clamp(0.0, 1.0)


def _sample_motion(
    *,
    cfg: DataConfig,
    direction: str,
    generator: torch.Generator,
) -> tuple[list[float], list[float]]:
    seq_len = int(cfg.seq_len)
    image_size = float(cfg.image_size)
    radius = 2.5
    speed = 2.0
    margin = radius + 2.0
    max_offset = speed * float(seq_len - 1)

    if direction == "right":
        start_x = float(torch.empty(1).uniform_(margin, image_size - margin - max_offset, generator=generator).item())
        start_y = float(torch.empty(1).uniform_(margin, image_size - margin, generator=generator).item())
        xs = [start_x + speed * float(t) for t in range(seq_len)]
        ys = [start_y for _ in range(seq_len)]
    elif direction == "left":
        start_x = float(torch.empty(1).uniform_(margin + max_offset, image_size - margin, generator=generator).item())
        start_y = float(torch.empty(1).uniform_(margin, image_size - margin, generator=generator).item())
        xs = [start_x - speed * float(t) for t in range(seq_len)]
        ys = [start_y for _ in range(seq_len)]
    elif direction == "down":
        start_x = float(torch.empty(1).uniform_(margin, image_size - margin, generator=generator).item())
        start_y = float(torch.empty(1).uniform_(margin, image_size - margin - max_offset, generator=generator).item())
        xs = [start_x for _ in range(seq_len)]
        ys = [start_y + speed * float(t) for t in range(seq_len)]
    elif direction == "up":
        start_x = float(torch.empty(1).uniform_(margin, image_size - margin, generator=generator).item())
        start_y = float(torch.empty(1).uniform_(margin + max_offset, image_size - margin, generator=generator).item())
        xs = [start_x for _ in range(seq_len)]
        ys = [start_y - speed * float(t) for t in range(seq_len)]
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    return xs, ys


def _make_prompt_answer(
    *,
    task_type: str,
    color_name: str,
    shape_name: str,
    direction: str,
) -> tuple[list[str], list[str]]:
    if task_type == "color":
        return ["what", "color", "is", "the", "object"], [color_name]
    if task_type == "shape":
        return ["what", "shape", "is", "the", "object"], [shape_name]
    if task_type in {"left", "right", "up", "down"}:
        return ["is", "it", "moving", task_type], ["yes" if direction == task_type else "no"]
    raise ValueError(f"Unsupported task_type: {task_type}")


def _build_record(
    *,
    cfg: DataConfig,
    idx: int,
    vocab: Vocab,
    generator: torch.Generator,
) -> dict[str, object]:
    color_name = tuple(_COLORS.keys())[int(torch.randint(0, len(_COLORS), (1,), generator=generator).item())]
    shape_name = _SHAPES[int(torch.randint(0, len(_SHAPES), (1,), generator=generator).item())]
    direction = _DIRECTIONS[int(torch.randint(0, len(_DIRECTIONS), (1,), generator=generator).item())]
    task_type = _TASK_TYPES[int(idx) % len(_TASK_TYPES)]

    prompt_tokens, answer_tokens = _make_prompt_answer(
        task_type=task_type,
        color_name=color_name,
        shape_name=shape_name,
        direction=direction,
    )
    prompt_ids = vocab.encode_prompt(prompt_tokens, max_length=int(cfg.max_text_length))
    input_ids, labels, attention_mask = vocab.encode_example(
        prompt_tokens,
        answer_tokens,
        max_length=int(cfg.max_text_length),
    )

    xs, ys = _sample_motion(cfg=cfg, direction=direction, generator=generator)
    frames = [
        _render_frame(
            image_size=int(cfg.image_size),
            color_name=color_name,
            shape_name=shape_name,
            center_x=xs[t],
            center_y=ys[t],
            radius=2.5,
            generator=generator,
        )
        for t in range(int(cfg.seq_len))
    ]

    return {
        "video": torch.stack(frames, dim=0),
        "prompt_ids": prompt_ids,
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "task_type": task_type,
        "prompt_text": " ".join(prompt_tokens),
        "answer_text": " ".join(answer_tokens),
    }


class SyntheticVideoVlmDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        self.cfg = cfg
        self.vocab = vocab
        if int(cfg.num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        if int(cfg.seq_len) < 3:
            raise ValueError("seq_len must be >= 3")
        if int(cfg.image_size) < 20:
            raise ValueError("image_size must be >= 20")
        if int(cfg.max_text_length) < 12:
            raise ValueError("max_text_length must be >= 12")

        generator = torch.Generator().manual_seed(int(cfg.seed))
        self.records = [
            _build_record(cfg=cfg, idx=idx, vocab=vocab, generator=generator)
            for idx in range(int(cfg.num_samples))
        ]

    def __len__(self) -> int:
        return int(len(self.records))

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.records[int(idx)]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Vocab]:
    vocab = _build_vocab()
    dataset = SyntheticVideoVlmDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "video": torch.stack([sample["video"] for sample in batch], dim=0),
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


__all__ = ["DataConfig", "SyntheticVideoVlmDataset", "Vocab", "get_dataloaders"]
