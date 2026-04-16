from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

LANDMARKS: tuple[str, ...] = ("left_eye", "right_eye", "nose_tip", "mouth_center", "chin_center")
LANDMARK_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(LANDMARKS)}


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
        "locate",
        "face",
        "landmark",
        "left_eye",
        "right_eye",
        "nose_tip",
        "mouth_center",
        "chin_center",
        "query",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _sample_face_landmarks(*, image_size: int, seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(int(seed))
    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)

    cx = float(rng.uniform(0.42, 0.58) * (image_size - 1))
    cy = float(rng.uniform(0.42, 0.58) * (image_size - 1))
    radius = float(rng.uniform(0.22, 0.30) * image_size)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    face = (dist <= radius).astype(np.float32)

    image = np.full((image_size, image_size), 0.08, dtype=np.float32)
    image += face * 0.54
    image += face * (0.10 * (1.0 - np.clip(dist / max(radius, 1e-6), 0.0, 1.0)))

    eye_dx = float(rng.uniform(0.28, 0.34) * radius)
    eye_y = cy - float(rng.uniform(0.12, 0.16) * radius)
    eye_sigma_x = float(max(1.0, 0.16 * radius))
    eye_sigma_y = float(max(1.0, 0.10 * radius))
    left_eye_x = cx - eye_dx
    right_eye_x = cx + eye_dx
    for eye_x in (left_eye_x, right_eye_x):
        eye = np.exp(-((xx - eye_x) ** 2) / (2.0 * eye_sigma_x**2) - ((yy - eye_y) ** 2) / (2.0 * eye_sigma_y**2))
        image -= 0.26 * eye

    nose_tip_x = cx + float(rng.normal(0.0, 0.04 * radius))
    nose_tip_y = cy + float(rng.uniform(0.08, 0.16) * radius)
    nose = np.exp(-((xx - nose_tip_x) ** 2) / (2.0 * 1.4**2) - ((yy - nose_tip_y) ** 2) / (2.0 * 2.0**2))
    image -= 0.10 * nose

    mouth_x = cx + float(rng.normal(0.0, 0.03 * radius))
    mouth_y = cy + float(rng.uniform(0.24, 0.32) * radius)
    mouth_w = float(max(2.0, 0.22 * radius))
    mouth_h = float(max(1.0, 0.08 * radius))
    mouth = np.exp(-((xx - mouth_x) ** 2) / (2.0 * mouth_w**2) - ((yy - mouth_y) ** 2) / (2.0 * mouth_h**2))
    image -= 0.13 * mouth

    chin_x = cx + float(rng.normal(0.0, 0.03 * radius))
    chin_y = cy + float(rng.uniform(0.38, 0.44) * radius)
    chin = np.exp(-((xx - chin_x) ** 2) / (2.0 * 2.2**2) - ((yy - chin_y) ** 2) / (2.0 * 1.6**2))
    image -= 0.07 * chin

    image += rng.normal(0.0, 0.04, size=(image_size, image_size)).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)

    landmarks = {
        "left_eye": np.array([left_eye_x, eye_y], dtype=np.float32),
        "right_eye": np.array([right_eye_x, eye_y], dtype=np.float32),
        "nose_tip": np.array([nose_tip_x, nose_tip_y], dtype=np.float32),
        "mouth_center": np.array([mouth_x, mouth_y], dtype=np.float32),
        "chin_center": np.array([chin_x, chin_y], dtype=np.float32),
    }
    return image, landmarks


class ToyFaceLandmarkReasoningDataset(Dataset):
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
        sample_seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 79
        image, landmarks = _sample_face_landmarks(image_size=int(self.cfg.image_size), seed=sample_seed)
        landmark_name = LANDMARKS[int(idx) % len(LANDMARKS)]
        query_ids, query_mask = self.vocab.encode_tokens(
            ["locate", "face", "landmark", landmark_name, "query"],
            max_length=int(self.cfg.max_text_length),
        )
        target_point = torch.from_numpy(landmarks[landmark_name] / float(self.cfg.image_size)).to(torch.float32)
        return {
            "image": torch.from_numpy(image).unsqueeze(0),
            "query_ids": query_ids,
            "query_mask": query_mask,
            "target_point": target_point,
            "landmark_name": landmark_name,
            "query_text": f"locate face landmark {landmark_name} query",
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    vocab = _build_vocab()
    dataset = ToyFaceLandmarkReasoningDataset(cfg, vocab=vocab)
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
    "LANDMARK_TO_ID",
    "LANDMARKS",
    "DataConfig",
    "ToyFaceLandmarkReasoningDataset",
    "Vocab",
    "get_dataloaders",
]
