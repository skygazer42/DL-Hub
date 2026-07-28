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
        "finger",
        "count",
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "query",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _segment_distance(
    xx: np.ndarray,
    yy: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    vx = float(end[0] - start[0])
    vy = float(end[1] - start[1])
    denom = vx * vx + vy * vy
    if denom <= 1e-6:
        return np.sqrt((xx - float(start[0])) ** 2 + (yy - float(start[1])) ** 2)

    proj = ((xx - float(start[0])) * vx + (yy - float(start[1])) * vy) / denom
    proj = np.clip(proj, 0.0, 1.0)
    closest_x = float(start[0]) + proj * vx
    closest_y = float(start[1]) + proj * vy
    return np.sqrt((xx - closest_x) ** 2 + (yy - closest_y) ** 2)


def _sample_finger_count(*, image_size: int, seed: int) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(int(seed))
    size = int(image_size)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    target = int(rng.integers(0, 6))

    image = np.full((size, size), 0.06, dtype=np.float32)
    image += 0.04 * (1.0 - yy / max(float(size - 1), 1.0))

    palm_cx = float(rng.uniform(0.46, 0.54) * (size - 1))
    palm_cy = float(rng.uniform(0.66, 0.76) * (size - 1))
    palm_r = float(rng.uniform(0.15, 0.19) * size)
    palm = np.exp(-((xx - palm_cx) ** 2 + (yy - palm_cy) ** 2) / (2.0 * palm_r * palm_r)).astype(np.float32)
    image += 0.28 * palm

    base_y = palm_cy - 0.55 * palm_r
    finger_len = float(rng.uniform(0.20, 0.28) * size)
    spread = float(rng.uniform(0.09, 0.13) * size)
    tilt = float(rng.uniform(-0.05, 0.05) * size)

    anchors = np.linspace(-2.0, 2.0, 5, dtype=np.float32) * spread
    for i in range(int(target)):
        x0 = palm_cx + float(anchors[i])
        start = np.array([x0, base_y], dtype=np.float32)
        end = np.array([x0 + 0.12 * anchors[i], base_y - finger_len + 0.18 * tilt], dtype=np.float32)
        np.clip(start, 2.0, float(size - 3), out=start)
        np.clip(end, 2.0, float(size - 3), out=end)

        dist = _segment_distance(xx, yy, start, end)
        stroke = np.exp(-(dist**2) / (2.0 * 1.25 * 1.25)).astype(np.float32)
        image += 0.24 * stroke

        tip = np.exp(-((xx - float(end[0])) ** 2 + (yy - float(end[1])) ** 2) / (2.0 * 1.7 * 1.7)).astype(
            np.float32
        )
        image += 0.12 * tip

    image += rng.normal(0.0, 0.02, size=image.shape).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)
    return image, target


class SyntheticFingerCountReasoningDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        if int(cfg.image_size) < 48:
            raise ValueError("image_size must be >= 48")
        if int(cfg.max_text_length) < 12:
            raise ValueError("max_text_length must be >= 12")
        self.cfg = cfg
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample_seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 191
        image, target = _sample_finger_count(image_size=int(self.cfg.image_size), seed=sample_seed)
        query_tokens = ["estimate", "finger", "count", "zero", "one", "two", "three", "four", "five", "query"]
        query_ids, query_mask = self.vocab.encode_tokens(query_tokens, max_length=int(self.cfg.max_text_length))
        return {
            "image": torch.from_numpy(image).to(torch.float32).unsqueeze(0),
            "query_ids": query_ids,
            "query_mask": query_mask,
            "target_finger_count": torch.tensor(int(target), dtype=torch.long),
            "query_text": "estimate finger count zero one two three four five query",
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    vocab = _build_vocab()
    dataset = SyntheticFingerCountReasoningDataset(cfg, vocab=vocab)
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
    "SyntheticFingerCountReasoningDataset",
    "Vocab",
    "get_dataloaders",
]

