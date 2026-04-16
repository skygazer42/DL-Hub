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
    max_text_length: int = 10
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    identity_pool: int = 24


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "verify",
        "face",
        "identity",
        "pair",
        "same",
        "different",
        "query",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _render_face(*, image_size: int, identity_id: int, variant_seed: int) -> torch.Tensor:
    id_rng = np.random.default_rng(int(identity_id) * 1_000_003 + 97)
    rng = np.random.default_rng(int(variant_seed))

    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    cx = float(id_rng.uniform(0.40, 0.60) * (image_size - 1))
    cy = float(id_rng.uniform(0.40, 0.60) * (image_size - 1))
    radius = float(id_rng.uniform(0.22, 0.31) * image_size)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    face = (dist <= radius).astype(np.float32)
    image = np.full((image_size, image_size), 0.10, dtype=np.float32)
    image += face * (0.52 + float(id_rng.uniform(-0.03, 0.03)))
    image += face * (0.08 * (1.0 - np.clip(dist / max(radius, 1e-6), 0.0, 1.0)))

    eye_dx = float(id_rng.uniform(0.30, 0.36) * radius)
    eye_y = cy - float(id_rng.uniform(0.12, 0.17) * radius)
    eye_sigma = float(id_rng.uniform(1.1, 1.6))
    for eye_x in (cx - eye_dx, cx + eye_dx):
        image -= 0.30 * np.exp(-((xx - eye_x) ** 2 + (yy - eye_y) ** 2) / (2.0 * eye_sigma * eye_sigma))

    nose_sigma = float(id_rng.uniform(1.2, 1.9))
    image += 0.08 * np.exp(-((xx - cx) ** 2 + (yy - (cy + 0.04 * radius)) ** 2) / (2.0 * nose_sigma * nose_sigma))
    mouth_w = float(id_rng.uniform(0.18, 0.26) * radius)
    mouth = np.exp(
        -((yy - (cy + 0.28 * radius)) ** 2) / (2.0 * 1.2 * 1.2) - ((xx - cx) ** 2) / (2.0 * mouth_w * mouth_w)
    )
    image -= 0.10 * mouth

    # Variant-specific factors keep same identity pairs correlated but not identical.
    image = np.roll(image, shift=int(rng.integers(-1, 2)), axis=1)
    image = np.roll(image, shift=int(rng.integers(-1, 2)), axis=0)
    image += float(rng.uniform(-0.03, 0.03))
    image += rng.normal(0.0, 0.04, size=(image_size, image_size)).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)
    return torch.from_numpy(image).unsqueeze(0)


class ToyFaceVerificationDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if int(cfg.max_text_length) < 7:
            raise ValueError("max_text_length must be >= 7")
        if int(cfg.identity_pool) < 2:
            raise ValueError("identity_pool must be >= 2")
        self.cfg = cfg
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample_seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 37
        rng = np.random.default_rng(sample_seed)
        same_identity = ((int(idx) + int(self.cfg.seed)) % 2) == 0

        identity_a = int(rng.integers(0, int(self.cfg.identity_pool)))
        if same_identity:
            identity_b = identity_a
            cue = "same"
        else:
            candidates = [i for i in range(int(self.cfg.identity_pool)) if i != identity_a]
            identity_b = int(candidates[int(rng.integers(0, len(candidates)))])
            cue = "different"

        image_a = _render_face(
            image_size=int(self.cfg.image_size),
            identity_id=identity_a,
            variant_seed=sample_seed + 11,
        )
        image_b = _render_face(
            image_size=int(self.cfg.image_size),
            identity_id=identity_b,
            variant_seed=sample_seed + 29,
        )
        query_ids, attention_mask = self.vocab.encode_tokens(
            ["verify", "face", "identity", "pair", cue, "query"],
            max_length=int(self.cfg.max_text_length),
        )
        return {
            "image_a": image_a,
            "image_b": image_b,
            "query_ids": query_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(1 if same_identity else 0, dtype=torch.long),
            "pair_text": cue,
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    vocab = _build_vocab()
    dataset = ToyFaceVerificationDataset(cfg, vocab=vocab)
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


__all__ = ["DataConfig", "ToyFaceVerificationDataset", "Vocab", "get_dataloaders"]
