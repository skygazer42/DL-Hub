from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


GESTURES = ("rest", "left_wave", "right_wave", "hands_up")
GESTURE_TO_ID = {name: idx for idx, name in enumerate(GESTURES)}


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
        "gesture",
        "hands_up",
        "left_wave",
        "right_wave",
        "rest",
        "estimate",
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


def _sample_gesture(*, image_size: int, seed: int) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(int(seed))
    size = int(image_size)

    # Two arm-control factors in [-1, 1] determine the gesture class.
    left_arm = float(rng.uniform(-1.0, 1.0))
    right_arm = float(rng.uniform(-1.0, 1.0))

    if left_arm > 0.55 and right_arm > 0.55:
        gesture = "hands_up"
    elif left_arm > 0.55 and right_arm < 0.15:
        gesture = "left_wave"
    elif right_arm > 0.55 and left_arm < 0.15:
        gesture = "right_wave"
    else:
        gesture = "rest"
    target = int(GESTURE_TO_ID[gesture])

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    image = np.full((size, size), 0.06, dtype=np.float32)
    image += 0.04 * (1.0 - yy / max(float(size - 1), 1.0))

    center_x = 0.50 * (size - 1)
    shoulder_y = 0.32 * (size - 1)
    shoulder_span = 0.22 * size
    torso_len = 0.24 * size
    arm_reach = 0.24 * size
    hand_drop = 0.16 * size

    left_shoulder = np.array([center_x - 0.5 * shoulder_span, shoulder_y], dtype=np.float32)
    right_shoulder = np.array([center_x + 0.5 * shoulder_span, shoulder_y], dtype=np.float32)
    pelvis = np.array([center_x, shoulder_y + torso_len], dtype=np.float32)

    # Arm lifts: higher arm factor => higher hands (smaller y).
    left_hand = left_shoulder + np.array(
        [-arm_reach, hand_drop - left_arm * 0.22 * size],
        dtype=np.float32,
    )
    right_hand = right_shoulder + np.array(
        [arm_reach, hand_drop - right_arm * 0.22 * size],
        dtype=np.float32,
    )

    points = np.stack([left_shoulder, right_shoulder, pelvis, left_hand, right_hand], axis=0)
    np.clip(points, 2.0, float(size - 3), out=points)

    segments = [
        (points[0], points[1], 0.16),
        (0.5 * (points[0] + points[1]), points[2], 0.22),
        (points[0], points[3], 0.22),
        (points[1], points[4], 0.22),
    ]
    for start, end, weight in segments:
        dist = _segment_distance(xx, yy, start, end)
        stroke = np.exp(-(dist**2) / (2.0 * 1.35 * 1.35)).astype(np.float32)
        image += weight * stroke

    for joint_idx, point in enumerate(points):
        sigma = 1.8 if joint_idx < 3 else 1.6
        joint = np.exp(
            -((xx - float(point[0])) ** 2 + (yy - float(point[1])) ** 2) / (2.0 * sigma * sigma)
        ).astype(np.float32)
        image += 0.14 * joint

    image += rng.normal(0.0, 0.02, size=image.shape).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)
    return image, target


class ToyGestureReasoningDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        if int(cfg.image_size) < 48:
            raise ValueError("image_size must be >= 48")
        if int(cfg.max_text_length) < 7:
            raise ValueError("max_text_length must be >= 7")
        self.cfg = cfg
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        sample_seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 173
        image, target = _sample_gesture(image_size=int(self.cfg.image_size), seed=sample_seed)
        query_ids, query_mask = self.vocab.encode_tokens(
            ["estimate", "gesture", "rest", "left_wave", "right_wave", "hands_up", "query"],
            max_length=int(self.cfg.max_text_length),
        )
        return {
            "image": torch.from_numpy(image).to(torch.float32).unsqueeze(0),
            "query_ids": query_ids,
            "query_mask": query_mask,
            "target_gesture": torch.tensor(int(target), dtype=torch.long),
            "query_text": "estimate gesture rest left_wave right_wave hands_up query",
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    vocab = _build_vocab()
    dataset = ToyGestureReasoningDataset(cfg, vocab=vocab)
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
    "GESTURES",
    "ToyGestureReasoningDataset",
    "Vocab",
    "get_dataloaders",
]

