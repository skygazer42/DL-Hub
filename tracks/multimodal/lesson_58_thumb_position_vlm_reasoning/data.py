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


CLASS_NAMES = ("low", "middle", "high")


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "estimate",
        "thumb",
        "position",
        "low",
        "middle",
        "high",
        "query",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _sample_hand_crop(*, image_size: int, seed: int) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(int(seed))
    size = int(image_size)
    label = int(rng.integers(0, len(CLASS_NAMES)))

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    image = np.full((size, size), 0.05, dtype=np.float32)
    image += 0.03 * (1.0 - yy / max(float(size - 1), 1.0))

    cx = float(rng.uniform(0.52, 0.58) * (size - 1))
    cy = float(rng.uniform(0.57, 0.64) * (size - 1))
    palm_rx = float(rng.uniform(0.14, 0.18) * size)
    palm_ry = float(rng.uniform(0.17, 0.22) * size)
    finger_w = float(rng.uniform(0.03, 0.045) * size)
    finger_h = float(rng.uniform(0.13, 0.18) * size)
    finger_spacing = float(rng.uniform(0.05, 0.08) * size)
    thumb_rx = float(rng.uniform(0.07, 0.09) * size)
    thumb_ry = float(rng.uniform(0.10, 0.13) * size)
    thumb_dx = float(rng.uniform(0.14, 0.18) * size)
    thumb_dy = float((-0.18, -0.02, 0.14)[label] * size + rng.uniform(-0.02, 0.02) * size)

    palm = ((xx - cx) / palm_rx) ** 2 + ((yy - cy) / palm_ry) ** 2
    image += 0.66 * np.exp(-palm).astype(np.float32)

    finger_base_y = cy - palm_ry * 1.08
    for offset in (-1.5, -0.5, 0.5, 1.5):
        fx = cx + offset * finger_spacing + float(rng.uniform(-0.01, 0.01) * size)
        fy = finger_base_y + float(rng.uniform(-0.02, 0.02) * size)
        finger = np.exp(-((xx - fx) ** 2) / (2.0 * finger_w * finger_w)).astype(np.float32)
        finger *= np.exp(-((yy - fy) ** 2) / (2.0 * finger_h * finger_h)).astype(np.float32)
        image += 0.32 * finger

    tx = cx - thumb_dx
    ty = cy + thumb_dy
    thumb = ((xx - tx) / thumb_rx) ** 2 + ((yy - ty) / thumb_ry) ** 2
    image += 0.56 * np.exp(-thumb).astype(np.float32)

    wrist_y = cy + palm_ry * 0.95
    image += 0.07 * np.exp(-((yy - wrist_y) ** 2) / (2.0 * 2.5 * 2.5)).astype(np.float32)

    image += rng.normal(0.0, 0.02, size=image.shape).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)
    return image, label


class SyntheticThumbPositionReasoningDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        if int(cfg.image_size) < 48:
            raise ValueError("image_size must be >= 48")
        if int(cfg.max_text_length) < 9:
            raise ValueError("max_text_length must be >= 9")
        self.cfg = cfg
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample_seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 193
        image, target = _sample_hand_crop(image_size=int(self.cfg.image_size), seed=sample_seed)
        query_ids, query_mask = self.vocab.encode_tokens(
            ["estimate", "thumb", "position", "low", "middle", "high", "query"],
            max_length=int(self.cfg.max_text_length),
        )
        return {
            "image": torch.from_numpy(image).to(torch.float32).unsqueeze(0),
            "query_ids": query_ids,
            "query_mask": query_mask,
            "target_thumb_position": torch.tensor(int(target), dtype=torch.long),
            "query_text": "estimate thumb position low middle high query",
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    vocab = _build_vocab()
    dataset = SyntheticThumbPositionReasoningDataset(cfg, vocab=vocab)
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
    "CLASS_NAMES",
    "DataConfig",
    "SyntheticThumbPositionReasoningDataset",
    "Vocab",
    "get_dataloaders",
]
