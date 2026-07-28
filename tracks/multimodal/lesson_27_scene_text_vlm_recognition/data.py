from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

_WORDS: tuple[str, ...] = ("alpha", "beta", "gamma", "delta")
_WORD_TO_ID: dict[str, int] = {word: idx for idx, word in enumerate(_WORDS)}
_WORD_COLORS: dict[str, tuple[float, float, float]] = {
    "alpha": (1.0, 0.22, 0.2),
    "beta": (0.2, 0.95, 0.25),
    "gamma": (0.2, 0.35, 1.0),
    "delta": (1.0, 0.85, 0.2),
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
            raise ValueError(
                f"Too many tokens for max_length={int(max_length)}: got {len(tokens)} tokens."
            )
        ids = [int(self.token_to_id[token]) for token in tokens]
        pad_count = int(max_length) - len(ids)
        ids.extend([self.pad_id] * pad_count)
        mask = [1.0] * (int(max_length) - pad_count) + [0.0] * pad_count
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
    image_size: int = 24
    max_text_length: int = 12
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "read",
        "the",
        "scene",
        "text",
        *_WORDS,
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _validate_config(cfg: DataConfig) -> None:
    if int(cfg.num_samples) <= 0:
        raise ValueError("num_samples must be positive")
    if int(cfg.image_size) < 16:
        raise ValueError("image_size must be >= 16")
    if int(cfg.max_text_length) < 5:
        raise ValueError("max_text_length must be >= 5")


def _render_scene_text_image(*, word: str, image_size: int, generator: torch.Generator) -> torch.Tensor:
    size = int(image_size)
    image = torch.full((3, size, size), 0.04, dtype=torch.float32)

    ys = torch.arange(size, dtype=torch.float32).view(-1, 1).expand(size, size)
    xs = torch.arange(size, dtype=torch.float32).view(1, -1).expand(size, size)
    offset = int(torch.randint(-2, 3, (1,), generator=generator).item())
    radius = max(3, size // 5)
    cy = float(size // 2 + offset)
    cx = float(size // 2 - offset)
    region = (ys - cy).pow(2) + (xs - cx).pow(2) <= float(radius * radius)

    color = torch.tensor(_WORD_COLORS[word], dtype=torch.float32).view(3, 1, 1)
    image = torch.where(region.unsqueeze(0), color.expand_as(image), image)

    # Add word-specific stripe pattern as a tiny OCR-like cue.
    word_id = int(_WORD_TO_ID[word])
    stripe_stride = 2 + word_id
    stripe = (xs.to(torch.long) % stripe_stride) == 0
    image = torch.where((region & stripe).unsqueeze(0), image * 0.55, image)

    image[:, 0, :] = 0.08
    image[:, -1, :] = 0.08
    image[:, :, 0] = 0.08
    image[:, :, -1] = 0.08
    return image.clamp(0.0, 1.0)


class SyntheticSceneTextDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        _validate_config(cfg)
        self.cfg = cfg
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        generator = torch.Generator().manual_seed(int(self.cfg.seed) * 997 + int(idx))
        word = _WORDS[int(idx) % len(_WORDS)]
        label_id = int(_WORD_TO_ID[word])
        prompt_tokens = ["read", "the", "scene", "text"]
        prompt_ids, prompt_mask = self.vocab.encode(
            prompt_tokens,
            max_length=int(self.cfg.max_text_length),
        )
        image = _render_scene_text_image(
            word=word,
            image_size=int(self.cfg.image_size),
            generator=generator,
        )

        return {
            "image": image,
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "label_ids": torch.tensor(label_id, dtype=torch.long),
            "scene_text": word,
            "prompt_text": " ".join(prompt_tokens),
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Vocab]:
    vocab = _build_vocab()
    dataset = SyntheticSceneTextDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "image": torch.stack([sample["image"] for sample in batch], dim=0),
            "prompt_ids": torch.stack([sample["prompt_ids"] for sample in batch], dim=0),
            "prompt_mask": torch.stack([sample["prompt_mask"] for sample in batch], dim=0),
            "label_ids": torch.stack([sample["label_ids"] for sample in batch], dim=0),
            "scene_text": [str(sample["scene_text"]) for sample in batch],
            "prompt_text": [str(sample["prompt_text"]) for sample in batch],
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


__all__ = ["DataConfig", "SyntheticSceneTextDataset", "Vocab", "get_dataloaders"]
