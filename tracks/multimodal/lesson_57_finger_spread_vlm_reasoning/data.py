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
        "spread",
        "angle",
        "query",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _sample_finger_spread(*, image_size: int, seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(int(seed))
    size = int(image_size)
    spread = float(rng.uniform(0.02, 0.98))

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    image = np.full((size, size), 0.05, dtype=np.float32)
    image += 0.03 * (1.0 - yy / max(float(size - 1), 1.0))

    cx = float(0.5 * (size - 1) + rng.uniform(-2.0, 2.0))
    cy = float(0.64 * (size - 1) + rng.uniform(-2.0, 2.0))
    palm = np.exp(
        -((xx - cx) ** 2) / (2.0 * (0.15 * size) ** 2) - ((yy - cy) ** 2) / (2.0 * (0.17 * size) ** 2)
    ).astype(np.float32)
    image += 0.56 * palm

    finger_sep = (0.035 + 0.09 * spread) * size
    finger_width = 0.026 * size
    finger_height = (0.14 + 0.04 * spread) * size
    finger_y = cy - 0.24 * size
    for offset in (-1.5, -0.5, 0.5, 1.5):
        fx = cx + offset * finger_sep + rng.uniform(-0.7, 0.7)
        fy = finger_y + rng.uniform(-0.8, 0.8)
        finger = np.exp(
            -((xx - fx) ** 2) / (2.0 * finger_width**2) - ((yy - fy) ** 2) / (2.0 * finger_height**2)
        ).astype(np.float32)
        image += 0.17 * finger

    thumb_cx = cx - (0.16 + 0.03 * spread) * size
    thumb_cy = cy - 0.01 * size
    thumb = np.exp(
        -((xx - thumb_cx) ** 2) / (2.0 * (0.075 * size) ** 2)
        - ((yy - thumb_cy) ** 2) / (2.0 * (0.045 * size) ** 2)
    ).astype(np.float32)
    image += 0.14 * thumb

    image += 0.05 * np.clip((yy - cy) / max(0.35 * size, 1.0), -1.0, 1.0)
    image += rng.normal(0.0, 0.02, size=(size, size)).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)
    return image, spread


class SyntheticFingerSpreadReasoningDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        if int(cfg.image_size) < 48:
            raise ValueError("image_size must be >= 48")
        if int(cfg.max_text_length) < 8:
            raise ValueError("max_text_length must be >= 8")
        self.cfg = cfg
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample_seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 223
        image, target = _sample_finger_spread(image_size=int(self.cfg.image_size), seed=sample_seed)
        query_ids, query_mask = self.vocab.encode_tokens(
            ["estimate", "finger", "spread", "angle", "query"],
            max_length=int(self.cfg.max_text_length),
        )
        return {
            "image": torch.from_numpy(image).to(torch.float32).unsqueeze(0),
            "query_ids": query_ids,
            "query_mask": query_mask,
            "target_finger_spread": torch.tensor([float(target)], dtype=torch.float32),
            "query_text": "estimate finger spread angle query",
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    vocab = _build_vocab()
    dataset = SyntheticFingerSpreadReasoningDataset(cfg, vocab=vocab)
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
    "SyntheticFingerSpreadReasoningDataset",
    "Vocab",
    "get_dataloaders",
]
