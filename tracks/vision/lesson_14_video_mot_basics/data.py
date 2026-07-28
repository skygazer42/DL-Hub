from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 32
    seq_len: int = 4
    image_size: int = 64
    max_objects: int = 3
    num_classes: int = 3
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0

    noise_std: float = 0.05
    min_box_size: int = 6
    max_box_size: int = 14
    max_speed: float = 2.5


def _bbox_from_center(*, cx: float, cy: float, w: float, h: float, image_size: int) -> tuple[int, int, int, int]:
    x1 = int(np.floor(float(cx) - 0.5 * float(w)))
    y1 = int(np.floor(float(cy) - 0.5 * float(h)))
    x2 = int(np.ceil(float(cx) + 0.5 * float(w)))
    y2 = int(np.ceil(float(cy) + 0.5 * float(h)))

    s = int(image_size)
    x1 = int(np.clip(x1, 0, s - 1))
    y1 = int(np.clip(y1, 0, s - 1))
    x2 = int(np.clip(x2, x1 + 1, s))
    y2 = int(np.clip(y2, y1 + 1, s))
    return x1, y1, x2, y2


class SyntheticVideoMOTDataset(Dataset):
    """Synthetic video MOT synthetic dataset.

    Output:
    - video: (T, C, H, W)
    - targets:
      - boxes: (M, 4) normalized xyxy for final frame
      - labels: (M,) class ids
      - present: (M,) 0/1 object existence mask
    where M = cfg.max_objects.
    """

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.seq_len) < 2:
            raise ValueError("seq_len must be >= 2")
        if int(cfg.image_size) < 16:
            raise ValueError("image_size must be >= 16")
        if int(cfg.max_objects) < 1:
            raise ValueError("max_objects must be >= 1")
        if int(cfg.num_classes) < 1:
            raise ValueError("num_classes must be >= 1")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if int(cfg.min_box_size) < 2 or int(cfg.max_box_size) < int(cfg.min_box_size):
            raise ValueError("invalid box size range")
        if float(cfg.noise_std) < 0.0:
            raise ValueError("noise_std must be >= 0")
        if float(cfg.max_speed) <= 0.0:
            raise ValueError("max_speed must be > 0")

        self.cfg = cfg
        self.base_seed = int(cfg.seed)
        self.palette = np.asarray(
            [
                [1.00, 0.35, 0.35],
                [0.35, 1.00, 0.35],
                [0.35, 0.60, 1.00],
                [1.00, 0.95, 0.35],
                [1.00, 0.40, 0.95],
            ],
            dtype=np.float32,
        )

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(self.base_seed + int(idx))

    def __getitem__(self, idx: int):
        rng = self._rng(int(idx))
        s = int(self.cfg.image_size)
        t = int(self.cfg.seq_len)
        m = int(self.cfg.max_objects)
        c = 3

        video = rng.normal(
            loc=0.0, scale=float(self.cfg.noise_std), size=(t, c, s, s)
        ).astype(np.float32)
        video = np.clip(video, -1.0, 1.0)

        boxes = np.zeros((m, 4), dtype=np.float32)
        labels = np.zeros((m,), dtype=np.int64)
        present = np.zeros((m,), dtype=np.float32)

        num_objects = int(rng.integers(1, m + 1))
        for slot in range(num_objects):
            label = int(rng.integers(0, int(self.cfg.num_classes)))
            w = float(rng.integers(int(self.cfg.min_box_size), int(self.cfg.max_box_size) + 1))
            h = float(rng.integers(int(self.cfg.min_box_size), int(self.cfg.max_box_size) + 1))

            margin_x = 0.5 * w + 1.0
            margin_y = 0.5 * h + 1.0
            cx0 = float(rng.uniform(margin_x, s - margin_x))
            cy0 = float(rng.uniform(margin_y, s - margin_y))
            vx = float(rng.uniform(-float(self.cfg.max_speed), float(self.cfg.max_speed)))
            vy = float(rng.uniform(-float(self.cfg.max_speed), float(self.cfg.max_speed)))
            if abs(vx) + abs(vy) < 0.5:
                vx += 0.75

            color = self.palette[label % len(self.palette)]

            last_box = (0, 0, 1, 1)
            for frame in range(t):
                cx = float(np.clip(cx0 + vx * float(frame), margin_x, s - margin_x))
                cy = float(np.clip(cy0 + vy * float(frame), margin_y, s - margin_y))
                x1, y1, x2, y2 = _bbox_from_center(cx=cx, cy=cy, w=w, h=h, image_size=s)

                patch = video[frame, :, y1:y2, x1:x2]
                video[frame, :, y1:y2, x1:x2] = np.maximum(patch, color[:, None, None])
                last_box = (x1, y1, x2, y2)

            x1, y1, x2, y2 = last_box
            boxes[slot] = np.asarray([x1 / s, y1 / s, x2 / s, y2 / s], dtype=np.float32)
            labels[slot] = label
            present[slot] = 1.0

        return (
            torch.from_numpy(video),  # (T, C, H, W)
            {
                "boxes": torch.from_numpy(boxes),  # (M, 4), normalized xyxy
                "labels": torch.from_numpy(labels),  # (M,)
                "present": torch.from_numpy(present),  # (M,)
            },
        )


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds = SyntheticVideoMOTDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(ds), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    def _collate(batch):
        videos = torch.stack([b[0] for b in batch], dim=0)
        boxes = torch.stack([b[1]["boxes"] for b in batch], dim=0)
        labels = torch.stack([b[1]["labels"] for b in batch], dim=0)
        present = torch.stack([b[1]["present"] for b in batch], dim=0)
        return videos, {"boxes": boxes, "labels": labels, "present": present}

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "SyntheticVideoMOTDataset", "get_dataloaders"]

