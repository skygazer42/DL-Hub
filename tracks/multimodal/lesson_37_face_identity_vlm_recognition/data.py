from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

IDENTITIES: tuple[str, ...] = ("alice", "bruno", "carla", "diego")
IDENTITY_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(IDENTITIES)}


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
    image_size: int = 40
    max_text_length: int = 10
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "identify",
        "face",
        "identity",
        "alice",
        "bruno",
        "carla",
        "diego",
        "person",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _render_identity_face(*, image_size: int, seed: int, identity_id: int) -> torch.Tensor:
    rng = np.random.default_rng(int(seed))
    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    cx = float(rng.uniform(0.44, 0.56) * (image_size - 1))
    cy = float(rng.uniform(0.43, 0.57) * (image_size - 1))
    radius = float(rng.uniform(0.23, 0.29) * image_size)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    face = (dist <= radius).astype(np.float32)
    image = np.full((image_size, image_size), 0.06, dtype=np.float32)
    image += face * 0.56
    image += face * (0.14 * (1.0 - np.clip(dist / max(radius, 1e-6), 0.0, 1.0)))

    eye_dx = 0.34 * radius
    eye_y = cy - 0.14 * radius
    for eye_x in (cx - eye_dx, cx + eye_dx):
        image -= 0.28 * np.exp(-((xx - eye_x) ** 2 + (yy - eye_y) ** 2) / (2.0 * 1.4 * 1.4))

    mouth = np.exp(
        -((yy - (cy + 0.30 * radius)) ** 2) / (2.0 * 1.0 * 1.0)
        - ((xx - cx) ** 2) / (2.0 * (0.24 * radius) ** 2)
    )
    image -= 0.12 * mouth

    if int(identity_id) == 0:
        image += 0.13 * np.exp(-((xx - cx) ** 2 + (yy - (cy + 0.02 * radius)) ** 2) / (2.0 * 2.2 * 2.2))
    elif int(identity_id) == 1:
        brow = np.exp(-((yy - (cy - 0.22 * radius)) ** 2) / (2.0 * 0.9 * 0.9))
        image -= 0.06 * brow * face
    elif int(identity_id) == 2:
        image += 0.08 * np.clip((xx - cx) / max(radius, 1.0), -1.0, 1.0) * face
    else:
        image += 0.08 * np.clip((cy - yy) / max(radius, 1.0), -1.0, 1.0) * face

    image += rng.normal(0.0, 0.045, size=(image_size, image_size)).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)
    return torch.from_numpy(image).unsqueeze(0)


class ToyFaceIdentityDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        if int(cfg.image_size) < 28:
            raise ValueError("image_size must be >= 28")
        if int(cfg.max_text_length) < 6:
            raise ValueError("max_text_length must be >= 6")
        self.cfg = cfg
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample_seed = int(self.cfg.seed) * 1_000_003 + int(idx)
        identity_id = (int(idx) + int(self.cfg.seed)) % len(IDENTITIES)
        identity_name = IDENTITIES[identity_id]
        identity_ids, identity_mask = self.vocab.encode_tokens(
            ["identify", "face", "identity", identity_name],
            max_length=int(self.cfg.max_text_length),
        )
        return {
            "image": _render_identity_face(
                image_size=int(self.cfg.image_size),
                seed=sample_seed,
                identity_id=identity_id,
            ),
            "identity_ids": identity_ids,
            "identity_mask": identity_mask,
            "labels": torch.tensor(identity_id, dtype=torch.long),
            "identity_name": identity_name,
            "identity_text": f"identify face identity {identity_name}",
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    vocab = _build_vocab()
    dataset = ToyFaceIdentityDataset(cfg, vocab=vocab)
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
    "IDENTITIES",
    "IDENTITY_TO_ID",
    "ToyFaceIdentityDataset",
    "Vocab",
    "get_dataloaders",
]

