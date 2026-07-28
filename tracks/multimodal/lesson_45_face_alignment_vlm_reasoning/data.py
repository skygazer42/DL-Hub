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


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "align",
        "face",
        "landmarks",
        "to",
        "canonical",
        "template",
        "query",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _make_synthetic_face(*, image_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)

    canonical = np.array(
        [
            [0.32, 0.38],
            [0.68, 0.38],
            [0.50, 0.54],
            [0.38, 0.70],
            [0.62, 0.70],
        ],
        dtype=np.float32,
    )

    center_shift = rng.normal(0.0, 0.03, size=(1, 2)).astype(np.float32)
    pose_jitter = rng.normal(0.0, 0.025, size=canonical.shape).astype(np.float32)
    posed_points = np.clip(canonical + center_shift + pose_jitter, 0.05, 0.95)
    target_points = canonical

    image = np.full((image_size, image_size), 0.08, dtype=np.float32)
    cx, cy = posed_points[:, 0].mean(), posed_points[:, 1].mean() + 0.03
    radius = 0.34
    dist = np.sqrt((xx / (image_size - 1) - cx) ** 2 + (yy / (image_size - 1) - cy) ** 2)
    face = (dist <= radius).astype(np.float32)
    image += face * 0.56

    for idx, (px, py) in enumerate(posed_points):
        sigma = 0.012 if idx < 2 else 0.010
        marker = np.exp(
            -((xx / (image_size - 1) - px) ** 2 + (yy / (image_size - 1) - py) ** 2) / (2.0 * sigma**2)
        )
        image -= 0.14 * marker

    image += rng.normal(0.0, 0.035, size=image.shape).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)
    return image, target_points.astype(np.float32)


class SyntheticFaceAlignmentReasoningDataset(Dataset):
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
        sample_seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 83
        image, target_points = _make_synthetic_face(image_size=int(self.cfg.image_size), seed=sample_seed)
        query_ids, query_mask = self.vocab.encode_tokens(
            ["align", "face", "landmarks", "to", "canonical", "template", "query"],
            max_length=int(self.cfg.max_text_length),
        )
        return {
            "image": torch.from_numpy(image).to(torch.float32).unsqueeze(0),
            "query_ids": query_ids,
            "query_mask": query_mask,
            "target_points": torch.from_numpy(target_points).to(torch.float32),
            "query_text": "align face landmarks to canonical template query",
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    vocab = _build_vocab()
    dataset = SyntheticFaceAlignmentReasoningDataset(cfg, vocab=vocab)
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
    "SyntheticFaceAlignmentReasoningDataset",
    "Vocab",
    "get_dataloaders",
]
