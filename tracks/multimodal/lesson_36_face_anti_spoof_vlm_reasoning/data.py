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


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "judge",
        "face",
        "authenticity",
        "authentic",
        "spoof",
        "moire",
        "screen",
        "artifact",
        "shadow",
        "live",
        "texture",
        "cue",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _render_face(*, image_size: int, seed: int, spoof: bool) -> torch.Tensor:
    rng = np.random.default_rng(int(seed))
    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)
    cx = float(rng.uniform(0.44, 0.56) * (image_size - 1))
    cy = float(rng.uniform(0.44, 0.56) * (image_size - 1))
    radius = float(rng.uniform(0.24, 0.30) * image_size)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    face = (dist <= radius).astype(np.float32)
    image = np.full((image_size, image_size), 0.08, dtype=np.float32)
    image += face * 0.58
    image += face * (0.10 * (1.0 - np.clip(dist / max(radius, 1e-6), 0.0, 1.0)))

    eye_dx = 0.34 * radius
    eye_y = cy - 0.14 * radius
    eye_sigma = 1.4
    for eye_x in (cx - eye_dx, cx + eye_dx):
        image -= 0.35 * np.exp(-((xx - eye_x) ** 2 + (yy - eye_y) ** 2) / (2.0 * eye_sigma * eye_sigma))

    image += 0.10 * np.exp(-((xx - cx) ** 2 + (yy - (cy + 0.06 * radius)) ** 2) / (2.0 * 1.6 * 1.6))
    mouth = np.exp(
        -((yy - (cy + 0.28 * radius)) ** 2) / (2.0 * 1.0 * 1.0)
        - ((xx - cx) ** 2) / (2.0 * (0.22 * radius) ** 2)
    )
    image -= 0.12 * mouth

    if spoof:
        border = np.zeros_like(image)
        border[[2, 3, image_size - 4, image_size - 3], :] = 1.0
        border[:, [2, 3, image_size - 4, image_size - 3]] = 1.0
        image += 0.18 * border
        stripes = (np.sin((xx + yy) * 0.55) > 0).astype(np.float32)
        image = 0.62 * image + 0.22 * stripes
        image = np.roll(image, shift=int(rng.integers(-1, 2)), axis=1)
    else:
        image += 0.08 * np.clip((xx - cx) / max(radius, 1.0), -1.0, 1.0) * face

    image += rng.normal(0.0, 0.05, size=(image_size, image_size)).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)
    return torch.from_numpy(image).unsqueeze(0)


class SyntheticDeepfakeReasoningDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if int(cfg.max_text_length) < 6:
            raise ValueError("max_text_length must be >= 6")
        self.cfg = cfg
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample_seed = int(self.cfg.seed) * 1_000_003 + int(idx)
        spoof = ((int(idx) + int(self.cfg.seed)) % 2) == 0
        cue_text = "moire" if spoof else "shadow"
        if spoof and (int(idx) % 3 == 0):
            cue_text = "screen"
        if spoof and (int(idx) % 5 == 0):
            cue_text = "artifact"
        if not spoof and (int(idx) % 4 == 0):
            cue_text = "live"
        authenticity_text = "spoof" if spoof else "authentic"
        query_ids, attention_mask = self.vocab.encode_tokens(
            ["judge", "face", "authenticity", "cue", cue_text],
            max_length=int(self.cfg.max_text_length),
        )
        return {
            "image": _render_face(image_size=int(self.cfg.image_size), seed=sample_seed, spoof=spoof),
            "query_ids": query_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(1 if spoof else 0, dtype=torch.long),
            "cue_text": cue_text,
            "authenticity_text": authenticity_text,
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    vocab = _build_vocab()
    dataset = SyntheticDeepfakeReasoningDataset(cfg, vocab=vocab)
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


__all__ = ["DataConfig", "SyntheticDeepfakeReasoningDataset", "Vocab", "get_dataloaders"]
