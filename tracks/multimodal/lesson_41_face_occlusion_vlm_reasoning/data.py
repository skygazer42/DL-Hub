from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


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
    heavy_threshold: float = 0.22


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "assess",
        "face",
        "occlusion",
        "level",
        "heavy",
        "light",
        "query",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _render_face_with_occlusion(*, image_size: int, sample_seed: int) -> tuple[torch.Tensor, float]:
    rng = np.random.default_rng(int(sample_seed))
    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)

    cx = float(rng.uniform(0.42, 0.58) * (image_size - 1))
    cy = float(rng.uniform(0.42, 0.58) * (image_size - 1))
    radius = float(rng.uniform(0.22, 0.30) * image_size)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    face = (dist <= radius).astype(np.float32)
    image = np.full((image_size, image_size), 0.09, dtype=np.float32)
    image += face * 0.54
    image += face * (0.10 * (1.0 - np.clip(dist / max(radius, 1e-6), 0.0, 1.0)))

    eye_dx = 0.32 * radius
    eye_y = cy - 0.15 * radius
    for eye_x in (cx - eye_dx, cx + eye_dx):
        image -= 0.26 * np.exp(-((xx - eye_x) ** 2 + (yy - eye_y) ** 2) / (2.0 * 1.3 * 1.3))

    mouth = np.exp(
        -((yy - (cy + 0.29 * radius)) ** 2) / (2.0 * 1.2 * 1.2)
        - ((xx - cx) ** 2) / (2.0 * (0.26 * radius) ** 2)
    )
    image -= 0.08 * mouth

    occluder = np.zeros((image_size, image_size), dtype=np.float32)
    mode = int(rng.integers(0, 3))
    if mode == 0:
        height = int(rng.integers(max(5, image_size // 10), max(6, image_size // 6)))
        y1 = int(np.clip(eye_y - height // 2, 0, image_size - 1))
        y2 = min(image_size, y1 + height)
        x1 = int(np.clip(cx - 0.7 * radius, 0, image_size - 1))
        x2 = int(np.clip(cx + 0.7 * radius, x1 + 1, image_size))
        occluder[y1:y2, x1:x2] = 1.0
    elif mode == 1:
        width = int(rng.integers(max(6, image_size // 7), max(7, image_size // 4)))
        height = int(rng.integers(max(8, image_size // 5), max(9, image_size // 3)))
        x1 = int(np.clip(cx + rng.uniform(-0.25, 0.1) * radius, 0, image_size - 1))
        y1 = int(np.clip(cy + rng.uniform(-0.05, 0.25) * radius, 0, image_size - 1))
        x2 = min(image_size, x1 + width)
        y2 = min(image_size, y1 + height)
        occluder[y1:y2, x1:x2] = 1.0
    else:
        width = int(rng.integers(max(8, image_size // 5), max(9, image_size // 3)))
        y_curve = cy + 0.12 * radius + 0.18 * radius * np.sin((xx - cx) / max(0.22 * radius, 1.0))
        band = (yy >= y_curve) & (yy <= (y_curve + width))
        occluder[band] = 1.0

    covered = occluder * face
    image = image * (1.0 - 0.80 * covered) + 0.05 * covered
    image += rng.normal(0.0, 0.04, size=(image_size, image_size)).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)

    face_area = max(float(face.sum()), 1.0)
    occlusion_ratio = float(covered.sum() / face_area)
    return torch.from_numpy(image).unsqueeze(0), occlusion_ratio


class SyntheticFaceOcclusionReasoningDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if int(cfg.max_text_length) < 8:
            raise ValueError("max_text_length must be >= 8")
        if not (0.0 < float(cfg.heavy_threshold) < 1.0):
            raise ValueError("heavy_threshold must be in (0, 1)")
        self.cfg = cfg
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample_seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 71
        image, occlusion_ratio = _render_face_with_occlusion(
            image_size=int(self.cfg.image_size),
            sample_seed=sample_seed,
        )
        is_heavy = int(occlusion_ratio >= float(self.cfg.heavy_threshold))
        cue = "heavy" if is_heavy else "light"
        query_ids, attention_mask = self.vocab.encode_tokens(
            ["assess", "face", "occlusion", "level", cue, "query"],
            max_length=int(self.cfg.max_text_length),
        )
        return {
            "image": image,
            "query_ids": query_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(is_heavy, dtype=torch.long),
            "occlusion_ratio": torch.tensor([occlusion_ratio], dtype=torch.float32),
            "occlusion_text": cue,
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    vocab = _build_vocab()
    dataset = SyntheticFaceOcclusionReasoningDataset(cfg, vocab=vocab)
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
    "DataConfig",
    "SyntheticFaceOcclusionReasoningDataset",
    "Vocab",
    "get_dataloaders",
]
