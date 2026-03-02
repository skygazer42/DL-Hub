from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 2048
    batch_size: int = 64
    image_size: int = 64
    stride: int = 4
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0

    noise_std: float = 0.15
    min_box_size: int = 10
    max_box_size: int = 28


class SyntheticRectDetection(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) % int(cfg.stride) != 0:
            raise ValueError("image_size must be divisible by stride")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if int(cfg.min_box_size) < 2 or int(cfg.max_box_size) < int(cfg.min_box_size):
            raise ValueError("invalid box size range")

        self.cfg = cfg
        self.grid_size = int(cfg.image_size) // int(cfg.stride)
        self.rng = np.random.default_rng(int(cfg.seed))

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _sample_box(self) -> tuple[int, int, int, int]:
        s = int(self.cfg.image_size)
        w = int(self.rng.integers(int(self.cfg.min_box_size), int(self.cfg.max_box_size) + 1))
        h = int(self.rng.integers(int(self.cfg.min_box_size), int(self.cfg.max_box_size) + 1))
        x1 = int(self.rng.integers(0, s - w))
        y1 = int(self.rng.integers(0, s - h))
        x2 = x1 + w
        y2 = y1 + h
        return x1, y1, x2, y2

    def __getitem__(self, idx: int):
        _ = int(idx)  # keep signature stable
        s = int(self.cfg.image_size)
        stride = int(self.cfg.stride)
        g = int(self.grid_size)

        x1, y1, x2, y2 = self._sample_box()

        # Image: background noise + bright rectangle.
        img = self.rng.normal(loc=0.0, scale=float(self.cfg.noise_std), size=(s, s)).astype(np.float32)
        img = np.clip(img, -1.0, 1.0)
        img[y1:y2, x1:x2] = 1.0

        # Compute center cell.
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        gcx = int(np.clip(int(cx // stride), 0, g - 1))
        gcy = int(np.clip(int(cy // stride), 0, g - 1))

        # Feature-map cell center in pixel coordinates.
        cell_x = (gcx + 0.5) * stride
        cell_y = (gcy + 0.5) * stride

        left = float(cell_x - x1)
        top = float(cell_y - y1)
        right = float(x2 - cell_x)
        bottom = float(y2 - cell_y)
        ltrb = np.array([left, top, right, bottom], dtype=np.float32)

        cls_target = np.zeros((1, g, g), dtype=np.float32)
        cls_target[0, gcy, gcx] = 1.0

        reg_target = np.zeros((4, g, g), dtype=np.float32)
        reg_target[:, gcy, gcx] = ltrb

        pos_mask = np.zeros((1, g, g), dtype=np.float32)
        pos_mask[0, gcy, gcx] = 1.0

        box = np.array([x1, y1, x2, y2], dtype=np.float32)

        return (
            torch.from_numpy(img).unsqueeze(0),  # (1, H, W)
            {
                "cls_target": torch.from_numpy(cls_target),  # (1, Gh, Gw)
                "reg_target": torch.from_numpy(reg_target),  # (4, Gh, Gw)
                "pos_mask": torch.from_numpy(pos_mask),  # (1, Gh, Gw)
                "box": torch.from_numpy(box),  # (4,)
            },
        )


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds = SyntheticRectDetection(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(ds), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    def _collate(batch):
        imgs = torch.stack([b[0] for b in batch], dim=0)
        cls_target = torch.stack([b[1]["cls_target"] for b in batch], dim=0)
        reg_target = torch.stack([b[1]["reg_target"] for b in batch], dim=0)
        pos_mask = torch.stack([b[1]["pos_mask"] for b in batch], dim=0)
        box = torch.stack([b[1]["box"] for b in batch], dim=0)
        return imgs, {"cls_target": cls_target, "reg_target": reg_target, "pos_mask": pos_mask, "box": box}

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


__all__ = ["DataConfig", "SyntheticRectDetection", "get_dataloaders"]
