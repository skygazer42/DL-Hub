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


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "estimate",
        "hand",
        "pose",
        "keypoints",
        "wrist",
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
        "query",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _segment_distance(
    xx: np.ndarray,
    yy: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    vx = float(end[0] - start[0])
    vy = float(end[1] - start[1])
    denom = vx * vx + vy * vy
    if denom <= 1e-6:
        return np.sqrt((xx - float(start[0])) ** 2 + (yy - float(start[1])) ** 2)

    proj = ((xx - float(start[0])) * vx + (yy - float(start[1])) * vy) / denom
    proj = np.clip(proj, 0.0, 1.0)
    closest_x = float(start[0]) + proj * vx
    closest_y = float(start[1]) + proj * vy
    return np.sqrt((xx - closest_x) ** 2 + (yy - closest_y) ** 2)


def _sample_hand_pose(*, image_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    size = int(image_size)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    center_x = float(rng.uniform(0.42, 0.58) * (size - 1))
    wrist_y = float(rng.uniform(0.66, 0.80) * (size - 1))
    spread = float(rng.uniform(0.08, 0.14) * size)
    palm_rise = float(rng.uniform(0.10, 0.16) * size)
    finger_len = float(rng.uniform(0.16, 0.24) * size)
    thumb_len = float(rng.uniform(0.14, 0.20) * size)
    tilt = float(rng.uniform(-0.08, 0.08) * size)

    wrist = np.array([center_x, wrist_y], dtype=np.float32)
    thumb_tip = wrist + np.array([-1.5 * spread, -0.35 * thumb_len + 0.35 * tilt], dtype=np.float32)

    index_base = wrist + np.array([-0.75 * spread, -0.45 * palm_rise], dtype=np.float32)
    index_tip = index_base + np.array([-0.18 * spread, -finger_len], dtype=np.float32)

    middle_base = wrist + np.array([-0.20 * spread, -0.62 * palm_rise], dtype=np.float32)
    middle_tip = middle_base + np.array([0.02 * spread, -1.10 * finger_len], dtype=np.float32)

    ring_base = wrist + np.array([0.35 * spread, -0.52 * palm_rise], dtype=np.float32)
    ring_tip = ring_base + np.array([0.12 * spread, -0.97 * finger_len], dtype=np.float32)

    pinky_base = wrist + np.array([0.86 * spread, -0.38 * palm_rise], dtype=np.float32)
    pinky_tip = pinky_base + np.array([0.28 * spread, -0.82 * finger_len], dtype=np.float32)

    points = np.stack(
        [
            wrist,
            thumb_tip,
            index_base,
            index_tip,
            middle_base,
            middle_tip,
            ring_base,
            ring_tip,
            pinky_base,
            pinky_tip,
        ],
        axis=0,
    ).astype(np.float32)
    np.clip(points, 2.0, float(size - 3), out=points)

    image = np.full((size, size), 0.06, dtype=np.float32)
    image += 0.04 * (1.0 - yy / max(float(size - 1), 1.0))

    segments = [
        (points[0], points[1], 0.20),
        (points[0], points[2], 0.18),
        (points[2], points[3], 0.22),
        (points[0], points[4], 0.18),
        (points[4], points[5], 0.24),
        (points[0], points[6], 0.18),
        (points[6], points[7], 0.22),
        (points[0], points[8], 0.17),
        (points[8], points[9], 0.20),
    ]
    for start, end, weight in segments:
        dist = _segment_distance(xx, yy, start, end)
        stroke = np.exp(-(dist**2) / (2.0 * 1.25 * 1.25)).astype(np.float32)
        image += weight * stroke

    for joint_idx, point in enumerate(points):
        sigma = 2.0 if joint_idx == 0 else 1.6
        joint = np.exp(
            -((xx - float(point[0])) ** 2 + (yy - float(point[1])) ** 2) / (2.0 * sigma * sigma)
        ).astype(np.float32)
        image += (0.16 if joint_idx == 0 else 0.13) * joint

    image += rng.normal(0.0, 0.02, size=image.shape).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)

    keypoints = (points / float(size - 1)).reshape(-1).astype(np.float32)
    return image, keypoints


class ToyHandPoseReasoningDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        if int(cfg.image_size) < 48:
            raise ValueError("image_size must be >= 48")
        if int(cfg.max_text_length) < 10:
            raise ValueError("max_text_length must be >= 10")
        self.cfg = cfg
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample_seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 149
        image, target_keypoints = _sample_hand_pose(image_size=int(self.cfg.image_size), seed=sample_seed)
        query_ids, query_mask = self.vocab.encode_tokens(
            ["estimate", "hand", "pose", "keypoints", "wrist", "thumb", "index", "middle", "ring", "pinky", "query"],
            max_length=int(self.cfg.max_text_length),
        )
        return {
            "image": torch.from_numpy(image).to(torch.float32).unsqueeze(0),
            "query_ids": query_ids,
            "query_mask": query_mask,
            "target_keypoints": torch.from_numpy(target_keypoints).to(torch.float32),
            "query_text": "estimate hand pose keypoints wrist thumb index middle ring pinky query",
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    vocab = _build_vocab()
    dataset = ToyHandPoseReasoningDataset(cfg, vocab=vocab)
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
    "DataConfig",
    "ToyHandPoseReasoningDataset",
    "Vocab",
    "get_dataloaders",
]

