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
_SIZES: tuple[str, ...] = ("small", "large")
_LOCATIONS: tuple[tuple[str, str], ...] = (
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
)
_TASK_TOKENS: tuple[str, ...] = ("dax", "blicket", "wug", "zup")
_ATTRIBUTES: tuple[str, ...] = ("color", "shape", "size", "location")


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    sep_token: str = "<sep>"
    image_token: str = "<image>"

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
    def image_token_id(self) -> int:
        return int(self.token_to_id[self.image_token])

    @property
    def size(self) -> int:
        return int(len(self.id_to_token))

    def encode_prompt(self, tokens: list[str], *, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = [self.bos_token, *tokens]
        token_ids = [int(self.token_to_id[token]) for token in seq]
        if len(token_ids) > int(max_length):
            raise ValueError(f"Prompt exceeds max_length={int(max_length)}.")
        pad_count = int(max_length) - len(token_ids)
        token_ids.extend([self.pad_id] * pad_count)
        mask = [1.0] * (int(max_length) - pad_count) + [0.0] * pad_count
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32),
        )

    def encode_example(
        self,
        prompt_tokens: list[str],
        answer_tokens: list[str],
        *,
        max_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_ids, _prompt_mask = self.encode_prompt(prompt_tokens, max_length=int(max_length))
        full_tokens = [self.bos_token, *prompt_tokens, *answer_tokens, self.eos_token]
        full_ids = [int(self.token_to_id[token]) for token in full_tokens]
        if len(full_ids) > int(max_length):
            raise ValueError(f"Example exceeds max_length={int(max_length)}.")

        labels = [-100] * (1 + len(prompt_tokens)) + [*[int(self.token_to_id[t]) for t in answer_tokens], self.eos_id]
        pad_count = int(max_length) - len(full_ids)
        full_ids.extend([self.pad_id] * pad_count)
        labels.extend([-100] * pad_count)
        mask = [1.0] * (int(max_length) - pad_count) + [0.0] * pad_count
        return (
            prompt_ids,
            torch.tensor(full_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32),
        )

    def decode_ids(self, ids: list[int]) -> list[str]:
        tokens: list[str] = []
        for idx in ids:
            token = self.id_to_token[int(idx)]
            if token in {self.pad_token, self.bos_token}:
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
            "image_token_id": int(self.image_token_id),
            "token_to_id": {k: int(v) for k, v in self.token_to_id.items()},
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 512
    batch_size: int = 32
    image_size: int = 16
    max_text_length: int = 28
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    num_shots: int = 2


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<sep>",
        "<image>",
        "example",
        "query",
        "what",
        "is",
        "dax",
        "blicket",
        "wug",
        "zup",
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
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _sample_concept(generator: torch.Generator) -> tuple[str, str, str, tuple[str, str]]:
    color_name = tuple(_COLORS.keys())[int(torch.randint(0, len(_COLORS), (1,), generator=generator).item())]
    shape_name = _SHAPES[int(torch.randint(0, len(_SHAPES), (1,), generator=generator).item())]
    size_name = _SIZES[int(torch.randint(0, len(_SIZES), (1,), generator=generator).item())]
    location = _LOCATIONS[int(torch.randint(0, len(_LOCATIONS), (1,), generator=generator).item())]
    return color_name, shape_name, size_name, location


def _answer_tokens(attribute_name: str, concept: tuple[str, str, str, tuple[str, str]]) -> list[str]:
    color_name, shape_name, size_name, location = concept
    if attribute_name == "color":
        return [color_name]
    if attribute_name == "shape":
        return [shape_name]
    if attribute_name == "size":
        return [size_name]
    if attribute_name == "location":
        return [location[0], location[1]]
    raise ValueError(f"Unsupported attribute: {attribute_name}")


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


def _sample_demo_set(
    *,
    generator: torch.Generator,
    attribute_name: str,
    num_shots: int,
) -> list[tuple[str, str, str, tuple[str, str]]]:
    demos: list[tuple[str, str, str, tuple[str, str]]] = []
    seen_answers: set[tuple[str, ...]] = set()
    while len(demos) < int(num_shots):
        concept = _sample_concept(generator)
        answer = tuple(_answer_tokens(attribute_name, concept))
        if answer in seen_answers:
            continue
        seen_answers.add(answer)
        demos.append(concept)
    return demos


def _build_record(
    *,
    cfg: DataConfig,
    idx: int,
    vocab: Vocab,
    generator: torch.Generator,
) -> dict[str, object]:
    task_token = _TASK_TOKENS[int(torch.randint(0, len(_TASK_TOKENS), (1,), generator=generator).item())]
    attribute_name = _ATTRIBUTES[int(torch.randint(0, len(_ATTRIBUTES), (1,), generator=generator).item())]
    support_concepts = _sample_demo_set(
        generator=generator,
        attribute_name=attribute_name,
        num_shots=int(cfg.num_shots),
    )
    query_concept = _sample_concept(generator)

    prompt_tokens: list[str] = []
    images: list[torch.Tensor] = []
    for concept in support_concepts:
        prompt_tokens.extend(
            [
                "example",
                vocab.image_token,
                "what",
                "is",
                task_token,
                vocab.sep_token,
                *_answer_tokens(attribute_name, concept),
                vocab.sep_token,
            ]
        )
        images.append(
            _render_shape(
                image_size=int(cfg.image_size),
                color_name=concept[0],
                shape_name=concept[1],
                size_name=concept[2],
                location=concept[3],
            )
        )

    prompt_tokens.extend(
        [
            "query",
            vocab.image_token,
            "what",
            "is",
            task_token,
            vocab.sep_token,
        ]
    )
    images.append(
        _render_shape(
            image_size=int(cfg.image_size),
            color_name=query_concept[0],
            shape_name=query_concept[1],
            size_name=query_concept[2],
            location=query_concept[3],
        )
    )

    answer_tokens = _answer_tokens(attribute_name, query_concept)
    prompt_ids, input_ids, labels, attention_mask = vocab.encode_example(
        prompt_tokens,
        answer_tokens,
        max_length=int(cfg.max_text_length),
    )

    return {
        "images": torch.stack(images, dim=0),
        "prompt_ids": prompt_ids,
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "task_token": task_token,
        "attribute_name": attribute_name,
        "prompt_text": " ".join(prompt_tokens),
        "answer_text": " ".join(answer_tokens),
        "record_id": int(idx),
    }


class ToyFlamingoDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        self.cfg = cfg
        self.vocab = vocab
        if int(cfg.num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        if int(cfg.image_size) < 12:
            raise ValueError("image_size must be >= 12")
        if int(cfg.max_text_length) < 28:
            raise ValueError("max_text_length must be >= 28")
        if int(cfg.num_shots) != 2:
            raise ValueError("num_shots must be 2 for this teaching lesson")

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
    dataset = ToyFlamingoDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "images": torch.stack([sample["images"] for sample in batch], dim=0),
            "prompt_ids": torch.stack([sample["prompt_ids"] for sample in batch], dim=0),
            "input_ids": torch.stack([sample["input_ids"] for sample in batch], dim=0),
            "labels": torch.stack([sample["labels"] for sample in batch], dim=0),
            "attention_mask": torch.stack([sample["attention_mask"] for sample in batch], dim=0),
            "task_token": [str(sample["task_token"]) for sample in batch],
            "attribute_name": [str(sample["attribute_name"]) for sample in batch],
            "prompt_text": [str(sample["prompt_text"]) for sample in batch],
            "answer_text": [str(sample["answer_text"]) for sample in batch],
            "record_id": torch.tensor([int(sample["record_id"]) for sample in batch], dtype=torch.long),
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


__all__ = ["DataConfig", "ToyFlamingoDataset", "Vocab", "get_dataloaders"]
