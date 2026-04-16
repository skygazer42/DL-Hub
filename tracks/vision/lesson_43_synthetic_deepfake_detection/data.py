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
    noise_std: float = 0.05


class SyntheticDeepfakeDetectionDataset:
    """Binary real-vs-deepfake synthetic face classification."""

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
        rng = np.random.default_rng(int(self.cfg.seed) * 1_000_003 + int(idx))
        label = int((int(idx) + int(self.cfg.seed)) % 2)

        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        cx = float(rng.uniform(0.44, 0.56) * (size - 1))
        cy = float(rng.uniform(0.43, 0.57) * (size - 1))
        radius = float(rng.uniform(0.24, 0.30) * size)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        face = (dist <= radius).astype(np.float32)
        image = np.full((size, size), 0.08, dtype=np.float32)
        image += face * 0.56
        image += face * (0.13 * (1.0 - np.clip(dist / max(radius, 1e-6), 0.0, 1.0)))

        eye_dx = 0.34 * radius
        eye_y = cy - 0.14 * radius
        eye_sigma = 1.5
        for eye_x in (cx - eye_dx, cx + eye_dx):
            image -= 0.34 * np.exp(
                -((xx - eye_x) ** 2 + (yy - eye_y) ** 2) / (2.0 * eye_sigma * eye_sigma)
            ).astype(np.float32)

        nose = np.exp(
            -((xx - cx) ** 2) / (2.0 * 1.8 * 1.8) - ((yy - (cy + 0.04 * radius)) ** 2) / (2.0 * 3.4 * 3.4)
        ).astype(np.float32)
        mouth = np.exp(
            -((yy - (cy + 0.28 * radius)) ** 2) / (2.0 * 1.0 * 1.0)
            - ((xx - cx) ** 2) / (2.0 * (0.22 * radius) ** 2)
        ).astype(np.float32)
        image += 0.08 * nose
        image -= 0.12 * mouth

        if label == 1:
            # Deepfake: blending seam, over-smoothing, and periodic generator artifacts.
            seam_radius = radius * float(rng.uniform(0.70, 0.86))
            seam = np.logical_and(dist >= seam_radius, dist <= seam_radius + 2.5).astype(np.float32)
            image += 0.16 * seam

            smooth = (
                image
                + np.roll(image, 1, axis=0)
                + np.roll(image, -1, axis=0)
                + np.roll(image, 1, axis=1)
                + np.roll(image, -1, axis=1)
            ) / 5.0
            image = 0.58 * smooth + 0.42 * image

            waves = 0.10 * np.sin(xx * 0.75 + yy * 0.18) + 0.05 * np.cos(yy * 0.82)
            image += waves.astype(np.float32) * face
            image = np.roll(image, shift=int(rng.integers(-2, 3)), axis=1)
        else:
            # Real: mild lighting asymmetry and sharper local contrast.
            image += 0.08 * np.clip((xx - cx) / max(radius, 1.0), -1.0, 1.0) * face
            image += 0.04 * np.clip((cy - yy) / max(radius, 1.0), -1.0, 1.0) * face

        image += rng.normal(0.0, float(self.cfg.noise_std), size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        return torch.from_numpy(image).unsqueeze(0), label


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticDeepfakeDetectionDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return images, labels

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


__all__ = ["DataConfig", "SyntheticDeepfakeDetectionDataset", "get_dataloaders"]
