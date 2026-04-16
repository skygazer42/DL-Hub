from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

IDENTITIES: tuple[str, ...] = ("atlas", "blair", "cyra", "dante", "eden")


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
        "retrieve",
        "face",
        "identity",
        "from",
        "gallery",
        "query",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _render_identity_face(*, image_size: int, identity_id: int, seed: int) -> np.ndarray:
    id_rng = np.random.default_rng(int(identity_id) * 1_000_003 + 211)
    rng = np.random.default_rng(int(seed))
    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)

    cx = float(id_rng.uniform(0.42, 0.58) * (image_size - 1)) + float(rng.uniform(-1.0, 1.0))
    cy = float(id_rng.uniform(0.43, 0.59) * (image_size - 1)) + float(rng.uniform(-1.0, 1.0))
    rx = float(id_rng.uniform(0.18, 0.27) * image_size)
    ry = float(id_rng.uniform(0.22, 0.30) * image_size)
    face_mask = (((xx - cx) / max(rx, 1e-6)) ** 2 + ((yy - cy) / max(ry, 1e-6)) ** 2 <= 1.0).astype(np.float32)

    image = np.full((image_size, image_size), 0.08, dtype=np.float32)
    image += 0.58 * face_mask
    image += 0.12 * face_mask * (1.0 - np.clip(np.sqrt(((xx - cx) / max(rx, 1e-6)) ** 2 + ((yy - cy) / max(ry, 1e-6)) ** 2), 0.0, 1.0))

    eye_dx_scale = (0.26, 0.34, 0.30, 0.22, 0.38)
    eye_y_scale = (0.18, 0.15, 0.20, 0.17, 0.13)
    mouth_width_scale = (0.22, 0.34, 0.28, 0.18, 0.30)
    mouth_shift = (-0.8, 0.6, 0.0, -1.2, 1.0)
    brow_strength = (0.04, 0.09, 0.06, 0.03, 0.08)
    nose_width_scale = (0.10, 0.15, 0.12, 0.08, 0.18)
    side_shading = (-0.06, 0.08, -0.02, 0.05, -0.08)

    eye_y = cy - eye_y_scale[identity_id] * ry
    eye_dx = eye_dx_scale[identity_id] * rx
    eye_sigma = 1.2 + 0.15 * identity_id
    for eye_x in (cx - eye_dx, cx + eye_dx):
        eye = np.exp(-((xx - eye_x) ** 2 + (yy - eye_y) ** 2) / (2.0 * eye_sigma * eye_sigma))
        image -= (0.28 + 0.01 * identity_id) * eye.astype(np.float32)

    brow_y = eye_y - 2.0
    brow_band = np.exp(-((yy - brow_y) ** 2) / (2.0 * 0.8 * 0.8)).astype(np.float32)
    brow_window = np.clip(1.0 - np.abs(xx - cx) / max(0.48 * rx, 1.0), 0.0, 1.0)
    image -= brow_strength[identity_id] * brow_band * brow_window

    nose_sigma_x = max(nose_width_scale[identity_id] * rx, 1.0)
    nose_sigma_y = max(0.16 * ry, 1.0)
    nose = np.exp(-((xx - cx) ** 2) / (2.0 * nose_sigma_x**2) - ((yy - cy) ** 2) / (2.0 * nose_sigma_y**2))
    image -= 0.09 * nose.astype(np.float32)

    mouth_center_y = cy + 0.29 * ry + mouth_shift[identity_id]
    mouth_width = max(mouth_width_scale[identity_id] * rx, 1.0)
    mouth = np.exp(
        -((yy - mouth_center_y) ** 2) / (2.0 * 1.0 * 1.0) - ((xx - cx) ** 2) / (2.0 * mouth_width**2)
    ).astype(np.float32)
    image -= (0.11 + 0.02 * identity_id) * mouth

    hairline = cy - (0.68 - 0.03 * identity_id) * ry
    image -= (0.10 + 0.01 * identity_id) * ((yy < hairline) * face_mask).astype(np.float32)
    image += side_shading[identity_id] * np.clip((xx - cx) / max(rx, 1.0), -1.0, 1.0) * face_mask

    image = np.roll(image, shift=int(rng.integers(-1, 2)), axis=0)
    image = np.roll(image, shift=int(rng.integers(-1, 2)), axis=1)
    image += rng.normal(0.0, 0.035, size=image.shape).astype(np.float32)
    return np.clip(image, 0.0, 1.0)


class ToyFaceRetrievalReasoningDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if int(cfg.max_text_length) < 7:
            raise ValueError("max_text_length must be >= 7")
        self.cfg = cfg
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample_seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 53
        identity_id = (int(idx) + int(self.cfg.seed)) % len(IDENTITIES)
        query_ids, query_mask = self.vocab.encode_tokens(
            ["retrieve", "face", "identity", "from", "gallery", "query"],
            max_length=int(self.cfg.max_text_length),
        )
        image = _render_identity_face(
            image_size=int(self.cfg.image_size),
            identity_id=identity_id,
            seed=sample_seed,
        )
        return {
            "image": torch.from_numpy(image).to(torch.float32).unsqueeze(0),
            "query_ids": query_ids,
            "query_mask": query_mask,
            "target_identity": torch.tensor(identity_id, dtype=torch.long),
            "query_text": "retrieve face identity from gallery query",
            "identity_name": IDENTITIES[identity_id],
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    vocab = _build_vocab()
    dataset = ToyFaceRetrievalReasoningDataset(cfg, vocab=vocab)
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
    "ToyFaceRetrievalReasoningDataset",
    "Vocab",
    "get_dataloaders",
]
