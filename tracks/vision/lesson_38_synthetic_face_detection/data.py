from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 2048
    batch_size: int = 32
    image_size: int = 48
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    in_channels: int = 1
    noise_std: float = 0.04


class SyntheticFaceDetectionDataset:
    """Synthetic single-face detection with one normalized box per image."""

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int):
        import torch

        size = int(self.cfg.image_size)
        rng = np.random.default_rng(int(self.cfg.seed) * 3_000_017 + int(idx))
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

        cx = float(rng.uniform(0.35, 0.65) * (size - 1))
        cy = float(rng.uniform(0.35, 0.65) * (size - 1))
        rx = float(rng.uniform(0.18, 0.28) * size)
        ry = float(rng.uniform(0.22, 0.30) * size)

        ellipse = (((xx - cx) / max(rx, 1e-6)) ** 2 + ((yy - cy) / max(ry, 1e-6)) ** 2) <= 1.0
        image = np.full((size, size), 0.08, dtype=np.float32)
        image[ellipse] = 0.70

        eye_y = cy - 0.18 * ry
        eye_dx = 0.34 * rx
        for eye_x in (cx - eye_dx, cx + eye_dx):
            eye = np.exp(-((xx - eye_x) ** 2 + (yy - eye_y) ** 2) / (2.0 * 1.4 * 1.4))
            image -= 0.30 * eye.astype(np.float32)

        mouth = np.exp(
            -((yy - (cy + 0.28 * ry)) ** 2) / (2.0 * 1.2 * 1.2)
            - ((xx - cx) ** 2) / (2.0 * (0.24 * rx) ** 2)
        )
        image -= 0.16 * mouth.astype(np.float32)

        image += 0.08 * np.clip((xx - cx) / max(rx, 1.0), -1.0, 1.0) * ellipse.astype(np.float32)
        image += rng.normal(0.0, float(self.cfg.noise_std), size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        x_min = np.clip((cx - rx) / max(size - 1, 1), 0.0, 1.0)
        y_min = np.clip((cy - ry) / max(size - 1, 1), 0.0, 1.0)
        x_max = np.clip((cx + rx) / max(size - 1, 1), 0.0, 1.0)
        y_max = np.clip((cy + ry) / max(size - 1, 1), 0.0, 1.0)
        box = np.asarray([x_min, y_min, x_max, y_max], dtype=np.float32)

        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(box)


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticFaceDetectionDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )

    def _collate(batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        boxes = torch.stack([item[1] for item in batch], dim=0)
        return images, boxes

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "SyntheticFaceDetectionDataset", "get_dataloaders"]
