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
    max_shift: float = 0.08
    noise_std: float = 0.04


class SyntheticFaceOcclusionDataset:
    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if int(cfg.in_channels) != 1:
            raise ValueError("This lesson expects grayscale inputs.")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int):
        import torch

        size = int(self.cfg.image_size)
        rng = np.random.default_rng(int(self.cfg.seed) * 97_409 + int(idx))

        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        cx = (0.5 + rng.uniform(-float(self.cfg.max_shift), float(self.cfg.max_shift))) * float(size - 1)
        cy = (0.52 + rng.uniform(-float(self.cfg.max_shift), float(self.cfg.max_shift))) * float(size - 1)
        radius_x = float(size) * rng.uniform(0.24, 0.31)
        radius_y = float(size) * rng.uniform(0.28, 0.35)

        face_mask = (((xx - cx) / max(radius_x, 1e-6)) ** 2 + ((yy - cy) / max(radius_y, 1e-6)) ** 2 <= 1.0)
        face_mask = face_mask.astype(np.float32)

        dist = np.sqrt(((xx - cx) / max(radius_x, 1e-6)) ** 2 + ((yy - cy) / max(radius_y, 1e-6)) ** 2)
        image = np.full((size, size), 0.08, dtype=np.float32)
        image += face_mask * 0.56
        image += face_mask * (0.1 * (1.0 - np.clip(dist, 0.0, 1.0)))

        eye_y = cy - 0.18 * radius_y
        eye_dx = 0.42 * radius_x
        for eye_x in (cx - eye_dx, cx + eye_dx):
            blob = np.exp(-((xx - eye_x) ** 2 + (yy - eye_y) ** 2) / (2.0 * 1.4 * 1.4)).astype(np.float32)
            image -= 0.35 * blob

        mouth_y = cy + 0.32 * radius_y
        mouth = np.exp(
            -((yy - mouth_y) ** 2) / (2.0 * 1.0 * 1.0)
            - ((xx - cx) ** 2) / (2.0 * (0.34 * radius_x) ** 2)
        ).astype(np.float32)
        image -= 0.08 * mouth

        occluder = np.zeros((size, size), dtype=np.float32)
        mode = int(rng.integers(0, 3))
        if mode == 0:
            height = int(rng.integers(max(3, size // 10), max(4, size // 6)))
            y1 = int(np.clip(eye_y - height // 2, 0, size - 1))
            y2 = min(size, y1 + height)
            x1 = int(np.clip(cx - 0.7 * radius_x, 0, size - 1))
            x2 = int(np.clip(cx + 0.7 * radius_x, x1 + 1, size))
            occluder[y1:y2, x1:x2] = 1.0
        elif mode == 1:
            width = int(rng.integers(max(4, size // 8), max(5, size // 5)))
            height = int(rng.integers(max(6, size // 6), max(7, size // 4)))
            x1 = int(np.clip(cx + rng.uniform(-0.3, 0.1) * radius_x, 0, size - 1))
            y1 = int(np.clip(cy + rng.uniform(-0.1, 0.25) * radius_y, 0, size - 1))
            x2 = min(size, x1 + width)
            y2 = min(size, y1 + height)
            occluder[y1:y2, x1:x2] = 1.0
        else:
            width = int(rng.integers(max(5, size // 6), max(6, size // 4)))
            height = int(rng.integers(max(5, size // 6), max(6, size // 4)))
            x1 = int(np.clip(cx - width // 2 + rng.uniform(-0.1, 0.1) * radius_x, 0, size - 1))
            y1 = int(np.clip(cy + rng.uniform(0.0, 0.25) * radius_y, 0, size - 1))
            x2 = min(size, x1 + width)
            y2 = min(size, y1 + height)
            occluder[y1:y2, x1:x2] = 1.0

        covered = occluder * face_mask
        image = image * (1.0 - 0.75 * covered) + 0.04 * covered
        image += rng.normal(0.0, float(self.cfg.noise_std), size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        face_area = max(float(face_mask.sum()), 1.0)
        occlusion_ratio = float(covered.sum() / face_area)

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        target_tensor = torch.tensor([occlusion_ratio], dtype=torch.float32)
        return image_tensor, target_tensor


def get_dataloaders(cfg: DataConfig):
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticFaceOcclusionDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "SyntheticFaceOcclusionDataset", "get_dataloaders"]
