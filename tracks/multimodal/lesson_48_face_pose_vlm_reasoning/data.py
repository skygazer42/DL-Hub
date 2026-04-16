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
    image_size: int = 48
    max_text_length: int = 14
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "estimate",
        "face",
        "pose",
        "yaw",
        "pitch",
        "roll",
        "query",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _rotate(offset_x: float, offset_y: float, *, angle: float) -> tuple[float, float]:
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    return cos_a * offset_x - sin_a * offset_y, sin_a * offset_x + cos_a * offset_y


def _sample_face_with_pose(*, image_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    yaw = float(rng.uniform(-1.0, 1.0))
    pitch = float(rng.uniform(-1.0, 1.0))
    roll = float(rng.uniform(-1.0, 1.0))

    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    x_norm = xx / float(max(image_size - 1, 1))
    y_norm = yy / float(max(image_size - 1, 1))

    cx = 0.50 + 0.05 * yaw
    cy = 0.55 + 0.04 * pitch
    rx = 0.22
    ry = 0.28

    face_mask = (((x_norm - cx) / rx) ** 2 + ((y_norm - cy) / ry) ** 2 <= 1.0).astype(np.float32)
    image = np.full((image_size, image_size), 0.07, dtype=np.float32)
    image += 0.60 * face_mask

    angle = roll * 0.55
    left_eye = _rotate(-0.26 * rx * (1.0 - 0.20 * yaw), -0.20 * ry - 0.05 * pitch, angle=angle)
    right_eye = _rotate(0.26 * rx * (1.0 + 0.20 * yaw), -0.20 * ry - 0.05 * pitch, angle=angle)
    for off_x, off_y in (left_eye, right_eye):
        eye = np.exp(-((x_norm - (cx + off_x)) ** 2 + (y_norm - (cy + off_y)) ** 2) / (2.0 * 0.012**2))
        image -= 0.22 * eye.astype(np.float32)

    nose_offset = _rotate(0.06 * yaw * rx, 0.02 * pitch * ry, angle=angle)
    nose = np.exp(
        -((x_norm - (cx + nose_offset[0])) ** 2) / (2.0 * 0.020**2)
        - ((y_norm - (cy + nose_offset[1])) ** 2) / (2.0 * 0.040**2)
    ).astype(np.float32)
    image -= 0.08 * nose

    mouth_offset = _rotate(0.08 * yaw * rx, 0.28 * ry + 0.05 * pitch * ry, angle=angle)
    mouth = np.exp(
        -((x_norm - (cx + mouth_offset[0])) ** 2) / (2.0 * (0.09 + 0.02 * abs(yaw)) ** 2)
        - ((y_norm - (cy + mouth_offset[1])) ** 2) / (2.0 * 0.018**2)
    ).astype(np.float32)
    image -= 0.10 * mouth

    image += 0.05 * yaw * np.clip((x_norm - cx) / rx, -1.0, 1.0) * face_mask
    image += rng.normal(0.0, 0.03, size=image.shape).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)
    target_pose = np.array([yaw, pitch, roll], dtype=np.float32)
    return image, target_pose


class ToyFacePoseReasoningDataset(Dataset):
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
        sample_seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 101
        image, target_pose = _sample_face_with_pose(image_size=int(self.cfg.image_size), seed=sample_seed)
        query_ids, query_mask = self.vocab.encode_tokens(
            ["estimate", "face", "pose", "yaw", "pitch", "roll", "query"],
            max_length=int(self.cfg.max_text_length),
        )
        return {
            "image": torch.from_numpy(image).to(torch.float32).unsqueeze(0),
            "query_ids": query_ids,
            "query_mask": query_mask,
            "target_pose": torch.from_numpy(target_pose).to(torch.float32),
            "query_text": "estimate face pose yaw pitch roll query",
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    vocab = _build_vocab()
    dataset = ToyFacePoseReasoningDataset(cfg, vocab=vocab)
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
    "ToyFacePoseReasoningDataset",
    "Vocab",
    "get_dataloaders",
]
