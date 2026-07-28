from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

REGIONS: tuple[str, ...] = ("eyes", "nose", "mouth", "chin")
REGION_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(REGIONS)}


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
        "region",
        "eyes",
        "nose",
        "mouth",
        "chin",
        "query",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _clip_box_xyxy(box: np.ndarray, *, image_size: int) -> np.ndarray:
    box = box.astype(np.float32)
    box[0] = float(np.clip(box[0], 0.0, float(image_size - 2)))
    box[1] = float(np.clip(box[1], 0.0, float(image_size - 2)))
    box[2] = float(np.clip(box[2], box[0] + 1.0, float(image_size - 1)))
    box[3] = float(np.clip(box[3], box[1] + 1.0, float(image_size - 1)))
    return box


def _sample_regions(*, image_size: int, seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(int(seed))
    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float32)

    cx = float(rng.uniform(0.42, 0.58) * (image_size - 1))
    cy = float(rng.uniform(0.42, 0.58) * (image_size - 1))
    radius = float(rng.uniform(0.22, 0.30) * image_size)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    face = (dist <= radius).astype(np.float32)

    image = np.full((image_size, image_size), 0.08, dtype=np.float32)
    image += face * 0.52
    image += face * (0.10 * (1.0 - np.clip(dist / max(radius, 1e-6), 0.0, 1.0)))

    eye_dx = float(rng.uniform(0.28, 0.34) * radius)
    eye_y = cy - float(rng.uniform(0.12, 0.16) * radius)
    eye_w = float(max(3.0, 0.18 * radius))
    eye_h = float(max(2.0, 0.12 * radius))
    for eye_x in (cx - eye_dx, cx + eye_dx):
        eye = np.exp(-((xx - eye_x) ** 2) / (2.0 * eye_w) - ((yy - eye_y) ** 2) / (2.0 * eye_h))
        image -= 0.24 * eye

    nose_x1 = cx - float(0.09 * radius)
    nose_y1 = cy - float(0.02 * radius)
    nose_x2 = cx + float(0.09 * radius)
    nose_y2 = cy + float(0.26 * radius)
    nose = np.exp(-((xx - cx) ** 2) / (2.0 * 1.4) - ((yy - (0.5 * (nose_y1 + nose_y2))) ** 2) / (2.0 * 4.0))
    image -= 0.10 * nose

    mouth_cy = cy + float(0.28 * radius)
    mouth_w = float(max(4.0, 0.24 * radius))
    mouth_h = float(max(2.0, 0.10 * radius))
    mouth = np.exp(-((yy - mouth_cy) ** 2) / (2.0 * mouth_h * mouth_h) - ((xx - cx) ** 2) / (2.0 * mouth_w * mouth_w))
    image -= 0.12 * mouth

    chin = np.clip((yy - (cy + 0.14 * radius)) / max(0.42 * radius, 1.0), 0.0, 1.0) * face
    image -= 0.08 * chin

    image += rng.normal(0.0, 0.04, size=(image_size, image_size)).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)

    boxes = {
        "eyes": _clip_box_xyxy(
            np.array(
                [
                    cx - eye_dx - 0.22 * radius,
                    eye_y - 0.14 * radius,
                    cx + eye_dx + 0.22 * radius,
                    eye_y + 0.14 * radius,
                ],
                dtype=np.float32,
            ),
            image_size=image_size,
        ),
        "nose": _clip_box_xyxy(np.array([nose_x1, nose_y1, nose_x2, nose_y2], dtype=np.float32), image_size=image_size),
        "mouth": _clip_box_xyxy(
            np.array(
                [
                    cx - 0.28 * radius,
                    mouth_cy - 0.12 * radius,
                    cx + 0.28 * radius,
                    mouth_cy + 0.12 * radius,
                ],
                dtype=np.float32,
            ),
            image_size=image_size,
        ),
        "chin": _clip_box_xyxy(
            np.array(
                [
                    cx - 0.24 * radius,
                    cy + 0.20 * radius,
                    cx + 0.24 * radius,
                    cy + 0.43 * radius,
                ],
                dtype=np.float32,
            ),
            image_size=image_size,
        ),
    }
    return image, boxes


class SyntheticFaceRegionDataset(Dataset):
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
        sample_seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 67
        image, region_boxes = _sample_regions(image_size=int(self.cfg.image_size), seed=sample_seed)
        region_name = REGIONS[int(idx) % len(REGIONS)]
        query_ids, query_mask = self.vocab.encode_tokens(
            ["locate", "face", "region", region_name, "query"],
            max_length=int(self.cfg.max_text_length),
        )
        box = torch.from_numpy(region_boxes[region_name] / float(self.cfg.image_size)).to(torch.float32)
        return {
            "image": torch.from_numpy(image).unsqueeze(0),
            "query_ids": query_ids,
            "query_mask": query_mask,
            "target_boxes": box,
            "region_name": region_name,
            "query_text": f"locate face region {region_name} query",
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    vocab = _build_vocab()
    dataset = SyntheticFaceRegionDataset(cfg, vocab=vocab)
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
    "REGION_TO_ID",
    "REGIONS",
    "DataConfig",
    "SyntheticFaceRegionDataset",
    "Vocab",
    "get_dataloaders",
]
