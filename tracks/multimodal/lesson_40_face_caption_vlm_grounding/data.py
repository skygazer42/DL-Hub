from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

ATTRIBUTES: tuple[str, ...] = ("smiling", "bearded", "glasses", "young")
ATTRIBUTE_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(ATTRIBUTES)}


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

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
    def size(self) -> int:
        return int(len(self.id_to_token))

    def encode_tokens(self, tokens: list[str], *, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = [self.bos_id, *[int(self.token_to_id[token]) for token in tokens], self.eos_id]
        if len(seq) > int(max_length):
            raise ValueError(f"Sequence exceeds max_length={int(max_length)}")
        pad_count = int(max_length) - len(seq)
        seq.extend([self.pad_id] * pad_count)
        mask = [1.0] * (int(max_length) - pad_count) + [0.0] * pad_count
        return torch.tensor(seq, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "pad_id": int(self.pad_id),
            "bos_id": int(self.bos_id),
            "eos_id": int(self.eos_id),
            "token_to_id": {k: int(v) for k, v in self.token_to_id.items()},
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 512
    batch_size: int = 16
    image_size: int = 48
    max_text_length: int = 12
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "face",
        "caption",
        "is",
        "smiling",
        "bearded",
        "glasses",
        "young",
        "match",
        "mismatch",
        "query",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _render_face(*, image_size: int, attribute_id: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(int(seed))
    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    cx = float(rng.uniform(0.42, 0.58) * (image_size - 1))
    cy = float(rng.uniform(0.42, 0.58) * (image_size - 1))
    radius = float(rng.uniform(0.22, 0.30) * image_size)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    face = (dist <= radius).astype(np.float32)
    image = np.full((image_size, image_size), 0.09, dtype=np.float32)
    image += face * 0.54
    image += face * (0.11 * (1.0 - np.clip(dist / max(radius, 1e-6), 0.0, 1.0)))

    eye_dx = 0.32 * radius
    eye_y = cy - 0.15 * radius
    for eye_x in (cx - eye_dx, cx + eye_dx):
        image -= 0.26 * np.exp(-((xx - eye_x) ** 2 + (yy - eye_y) ** 2) / (2.0 * 1.3 * 1.3))

    mouth = np.exp(
        -((yy - (cy + 0.29 * radius)) ** 2) / (2.0 * 1.2 * 1.2)
        - ((xx - cx) ** 2) / (2.0 * (0.26 * radius) ** 2)
    )
    image -= 0.08 * mouth

    if int(attribute_id) == 0:  # smiling
        image += 0.10 * np.exp(
            -((yy - (cy + 0.32 * radius)) ** 2) / (2.0 * 0.8 * 0.8)
            - ((xx - cx) ** 2) / (2.0 * (0.30 * radius) ** 2)
        )
    elif int(attribute_id) == 1:  # bearded
        beard = np.clip((yy - (cy + 0.10 * radius)) / max(0.35 * radius, 1.0), 0.0, 1.0) * face
        image -= 0.12 * beard
    elif int(attribute_id) == 2:  # glasses
        for x0 in (cx - eye_dx, cx + eye_dx):
            ring = np.exp(-((np.sqrt((xx - x0) ** 2 + (yy - eye_y) ** 2) - 2.8) ** 2) / (2.0 * 0.8 * 0.8))
            image -= 0.12 * ring
    else:  # young
        image += 0.07 * np.exp(-((xx - cx) ** 2 + (yy - (cy - 0.03 * radius)) ** 2) / (2.0 * 2.0 * 2.0))

    image += rng.normal(0.0, 0.04, size=(image_size, image_size)).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)
    return torch.from_numpy(image).unsqueeze(0)


class SyntheticFaceCaptionDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if int(cfg.max_text_length) < 8:
            raise ValueError("max_text_length must be >= 8")
        self.cfg = cfg
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample_seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 53
        true_attr_id = (int(idx) + int(self.cfg.seed)) % len(ATTRIBUTES)
        is_match = ((int(idx) + int(self.cfg.seed)) % 2) == 0

        if is_match:
            caption_attr_id = true_attr_id
            cue = "match"
            label = 1
        else:
            caption_attr_id = (true_attr_id + 1 + (int(idx) % (len(ATTRIBUTES) - 1))) % len(ATTRIBUTES)
            cue = "mismatch"
            label = 0

        caption_attr = ATTRIBUTES[caption_attr_id]
        caption_ids, caption_mask = self.vocab.encode_tokens(
            ["face", "caption", "is", caption_attr, cue, "query"],
            max_length=int(self.cfg.max_text_length),
        )
        return {
            "image": _render_face(
                image_size=int(self.cfg.image_size),
                attribute_id=true_attr_id,
                seed=sample_seed,
            ),
            "caption_ids": caption_ids,
            "caption_mask": caption_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "caption_text": f"face caption is {caption_attr} {cue} query",
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    vocab = _build_vocab()
    dataset = SyntheticFaceCaptionDataset(cfg, vocab=vocab)
    train_indices, val_indices = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )
    return train_loader, val_loader, vocab


__all__ = [
    "ATTRIBUTE_TO_ID",
    "ATTRIBUTES",
    "DataConfig",
    "SyntheticFaceCaptionDataset",
    "Vocab",
    "get_dataloaders",
]
