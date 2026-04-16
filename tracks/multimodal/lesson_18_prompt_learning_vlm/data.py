from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

_CONCEPTS: tuple[tuple[str, str], ...] = (
    ("crimson", "square"),
    ("amber", "triangle"),
    ("teal", "circle"),
    ("violet", "star"),
)
_COLORS: dict[str, tuple[float, float, float]] = {
    "crimson": (0.92, 0.18, 0.24),
    "amber": (0.98, 0.72, 0.18),
    "teal": (0.12, 0.75, 0.72),
    "violet": (0.62, 0.32, 0.86),
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

    def encode_text(self, tokens: list[str], *, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if len(tokens) > int(max_length):
            raise ValueError(f"Text exceeds max_length={int(max_length)}")
        ids = [int(self.token_to_id[token]) for token in tokens]
        pad_count = int(max_length) - len(ids)
        ids.extend([self.pad_id] * pad_count)
        mask = [1.0] * len(tokens) + [0.0] * pad_count
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "pad_id": int(self.pad_id),
            "token_to_id": {key: int(value) for key, value in self.token_to_id.items()},
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 256
    batch_size: int = 16
    image_size: int = 16
    max_text_length: int = 8
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def build_vocab() -> Vocab:
    tokens = ["<pad>", "crimson", "amber", "teal", "violet", "square", "triangle", "circle", "star"]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _render_concept_image(*, concept_idx: int, image_size: int, generator: torch.Generator) -> torch.Tensor:
    color_name, shape_name = _CONCEPTS[int(concept_idx)]
    color = torch.tensor(_COLORS[color_name], dtype=torch.float32).view(3, 1, 1)
    image = torch.full((3, int(image_size), int(image_size)), 0.04, dtype=torch.float32)
    noise = torch.rand((3, int(image_size), int(image_size)), generator=generator, dtype=torch.float32) * 0.03
    image = image + noise

    ys = torch.arange(int(image_size), dtype=torch.float32).view(-1, 1)
    xs = torch.arange(int(image_size), dtype=torch.float32).view(1, -1)
    yy = ys.expand(int(image_size), int(image_size))
    xx = xs.expand(int(image_size), int(image_size))
    center = (int(image_size) - 1) / 2.0
    radius = max(2.0, float(image_size) * 0.24)

    if shape_name == "square":
        mask = (yy - center).abs() <= radius
        mask = mask & ((xx - center).abs() <= radius)
    elif shape_name == "triangle":
        top = center - radius
        height = radius * 1.9
        lower = yy >= top
        upper = yy <= (top + height)
        spread = (yy - top) / height * radius
        mask = lower & upper & ((xx - center).abs() <= spread)
    elif shape_name == "circle":
        mask = (yy - center).pow(2) + (xx - center).pow(2) <= radius * radius
    elif shape_name == "star":
        vertical = (xx - center).abs() <= 1.0
        horizontal = (yy - center).abs() <= 1.0
        diag_a = ((yy - center) - (xx - center)).abs() <= 1.2
        diag_b = ((yy - center) + (xx - center)).abs() <= 1.2
        reach = ((yy - center).abs() <= radius) & ((xx - center).abs() <= radius)
        mask = reach & (vertical | horizontal | diag_a | diag_b)
    else:
        raise ValueError(f"Unsupported shape: {shape_name}")

    image = torch.where(mask.unsqueeze(0), color.expand_as(image), image)
    image[:, 0, :] = 0.08
    image[:, -1, :] = 0.08
    image[:, :, 0] = 0.08
    image[:, :, -1] = 0.08
    return image.clamp(0.0, 1.0)


class PromptLearningDataset(Dataset[dict[str, object]]):
    def __init__(self, cfg: DataConfig, vocab: Vocab) -> None:
        self.cfg = cfg
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        concept_idx = int(index) % len(_CONCEPTS)
        color_name, shape_name = _CONCEPTS[concept_idx]
        text_tokens = [color_name, shape_name]
        input_ids, attention_mask = self.vocab.encode_text(
            text_tokens,
            max_length=int(self.cfg.max_text_length),
        )
        generator = torch.Generator().manual_seed(int(self.cfg.seed) * 10_000 + int(index))
        image = _render_concept_image(
            concept_idx=concept_idx,
            image_size=int(self.cfg.image_size),
            generator=generator,
        )
        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_index": torch.tensor(concept_idx, dtype=torch.long),
            "concept_name": f"{color_name} {shape_name}",
        }


def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
        "target_index": torch.stack([item["target_index"] for item in batch], dim=0),
        "concept_name": [str(item["concept_name"]) for item in batch],
    }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    vocab = build_vocab()
    dataset = PromptLearningDataset(cfg, vocab)
    train_indices, val_indices = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    return train_loader, val_loader, vocab


__all__ = ["DataConfig", "PromptLearningDataset", "Vocab", "build_vocab", "get_dataloaders"]
