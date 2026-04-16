from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

PARTS: tuple[str, ...] = ("eyes", "mouth", "hair", "skin")
PART_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(PARTS)}


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
        "parse",
        "face",
        "part",
        "mask",
        "eyes",
        "mouth",
        "hair",
        "skin",
        "query",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _sample_face_parts(*, image_size: int, seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(int(seed))
    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)

    cx = float(rng.uniform(0.42, 0.58) * (image_size - 1))
    cy = float(rng.uniform(0.43, 0.60) * (image_size - 1))
    radius = float(rng.uniform(0.21, 0.29) * image_size)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    face = (dist <= radius).astype(np.float32)

    eye_dx = float(rng.uniform(0.27, 0.34) * radius)
    eye_y = cy - float(rng.uniform(0.12, 0.17) * radius)
    eye_rx = float(max(2.0, 0.11 * radius))
    eye_ry = float(max(1.5, 0.08 * radius))
    left_eye = (((xx - (cx - eye_dx)) / eye_rx) ** 2 + ((yy - eye_y) / eye_ry) ** 2 <= 1.0).astype(np.float32)
    right_eye = (((xx - (cx + eye_dx)) / eye_rx) ** 2 + ((yy - eye_y) / eye_ry) ** 2 <= 1.0).astype(np.float32)
    eyes = np.clip(left_eye + right_eye, 0.0, 1.0) * face

    mouth_y = cy + float(0.28 * radius)
    mouth_rx = float(max(3.0, 0.24 * radius))
    mouth_ry = float(max(1.5, 0.09 * radius))
    mouth = (((xx - cx) / mouth_rx) ** 2 + ((yy - mouth_y) / mouth_ry) ** 2 <= 1.0).astype(np.float32) * face

    hair_center_y = cy - float(0.62 * radius)
    hair_rx = float(1.10 * radius)
    hair_ry = float(0.56 * radius)
    hair = (((xx - cx) / hair_rx) ** 2 + ((yy - hair_center_y) / hair_ry) ** 2 <= 1.0).astype(np.float32)
    hair = hair * (yy <= cy - 0.02 * radius)

    skin = np.clip(face - eyes - mouth, 0.0, 1.0)
    skin = np.clip(skin * (1.0 - 0.35 * hair), 0.0, 1.0)

    image = np.full((image_size, image_size), 0.08, dtype=np.float32)
    image += face * 0.36
    image += skin * 0.16
    image += hair * 0.26
    image -= eyes * 0.22
    image -= mouth * 0.12
    image += rng.normal(0.0, 0.035, size=(image_size, image_size)).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)

    masks = {
        "eyes": eyes,
        "mouth": mouth,
        "hair": np.clip(hair, 0.0, 1.0),
        "skin": np.clip(skin, 0.0, 1.0),
    }
    return image, masks


class ToyFaceParsingDataset(Dataset):
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
        sample_seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 73
        image, masks = _sample_face_parts(image_size=int(self.cfg.image_size), seed=sample_seed)
        part_name = PARTS[int(idx) % len(PARTS)]
        query_ids, query_mask = self.vocab.encode_tokens(
            ["parse", "face", "part", part_name, "mask", "query"],
            max_length=int(self.cfg.max_text_length),
        )
        target_mask = torch.from_numpy(masks[part_name]).to(torch.float32).unsqueeze(0)
        return {
            "image": torch.from_numpy(image).to(torch.float32).unsqueeze(0),
            "query_ids": query_ids,
            "query_mask": query_mask,
            "target_mask": target_mask,
            "part_name": part_name,
            "query_text": f"parse face part {part_name} mask query",
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    vocab = _build_vocab()
    dataset = ToyFaceParsingDataset(cfg, vocab=vocab)
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
    "PART_TO_ID",
    "PARTS",
    "DataConfig",
    "ToyFaceParsingDataset",
    "Vocab",
    "get_dataloaders",
]
