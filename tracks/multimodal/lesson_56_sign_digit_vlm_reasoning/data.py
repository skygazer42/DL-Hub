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
            "token_to_id": {token: int(idx) for token, idx in self.token_to_id.items()},
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 512
    batch_size: int = 16
    image_size: int = 64
    max_text_length: int = 16
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "estimate",
        "sign",
        "digit",
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "query",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _sample_sign_digit(*, image_size: int, seed: int) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(int(seed))
    size = int(image_size)
    label = int(rng.integers(0, 10))

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    image = np.full((size, size), 0.05, dtype=np.float32)
    image += 0.03 * (1.0 - yy / max(float(size - 1), 1.0))

    cx = float(rng.uniform(0.46, 0.54) * (size - 1))
    cy = float(rng.uniform(0.58, 0.66) * (size - 1))
    palm_rx = float(rng.uniform(0.14, 0.18) * size)
    palm_ry = float(rng.uniform(0.16, 0.22) * size)
    palm = ((xx - cx) / palm_rx) ** 2 + ((yy - cy) / palm_ry) ** 2
    image += 0.68 * np.exp(-palm).astype(np.float32)

    finger_count = int(label % 5)
    style_group = int(label // 5)
    finger_base_y = cy - palm_ry * 1.05
    finger_w = float(rng.uniform(0.035, 0.05) * size)
    finger_h = float(rng.uniform(0.14, 0.22) * size)
    finger_spacing = float(rng.uniform(0.07, 0.09) * size)
    if finger_count > 0:
        offset_center = 0.5 * float(finger_count - 1)
        for k in range(finger_count):
            x_offset = (float(k) - offset_center) * finger_spacing
            fx = cx + x_offset + float(rng.uniform(-0.02, 0.02) * size)
            fy = finger_base_y - float(rng.uniform(0.10, 0.35) * finger_h)
            w = finger_w * float(rng.uniform(0.9, 1.1))
            h = finger_h * float(rng.uniform(0.9, 1.1))
            finger = np.exp(-((xx - fx) ** 2) / (2.0 * w * w)).astype(np.float32)
            finger *= np.exp(-((yy - fy) ** 2) / (2.0 * h * h)).astype(np.float32)
            image += 0.55 * finger

    slot = int(label % 5)
    mx = float(rng.uniform(0.08, 0.14) * size) if style_group == 0 else float(rng.uniform(0.86, 0.92) * size)
    my = float((0.12 + 0.14 * slot) * size)
    marker = np.exp(-((xx - mx) ** 2 + (yy - my) ** 2) / (2.0 * 1.4 * 1.4)).astype(np.float32)
    image += 0.22 * marker

    wrist_y = cy + palm_ry * 0.9
    image += 0.08 * np.exp(-((yy - wrist_y) ** 2) / (2.0 * 2.4 * 2.4)).astype(np.float32)

    image += rng.normal(0.0, 0.02, size=(size, size)).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)
    return image, label


class SyntheticSignDigitReasoningDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        if int(cfg.image_size) < 48:
            raise ValueError("image_size must be >= 48")
        if int(cfg.max_text_length) < 16:
            raise ValueError("max_text_length must be >= 16")
        self.cfg = cfg
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample_seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 181
        image, target = _sample_sign_digit(image_size=int(self.cfg.image_size), seed=sample_seed)
        query_ids, query_mask = self.vocab.encode_tokens(
            [
                "estimate",
                "sign",
                "digit",
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "query",
            ],
            max_length=int(self.cfg.max_text_length),
        )
        return {
            "image": torch.from_numpy(image).to(torch.float32).unsqueeze(0),
            "query_ids": query_ids,
            "query_mask": query_mask,
            "target_sign_digit": torch.tensor(int(target), dtype=torch.long),
            "query_text": "estimate sign digit zero one two three four five six seven eight nine query",
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    vocab = _build_vocab()
    dataset = SyntheticSignDigitReasoningDataset(cfg, vocab=vocab)
    train_indices, val_indices = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )
    return train_loader, val_loader, vocab


__all__ = [
    "DataConfig",
    "SyntheticSignDigitReasoningDataset",
    "Vocab",
    "get_dataloaders",
]
