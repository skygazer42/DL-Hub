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


class SyntheticFaceLivenessDataset:
    """Binary live-vs-spoof synthetic face classification."""

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
        rng = np.random.default_rng(int(self.cfg.seed) * 2_000_003 + int(idx))
        label = int((int(idx) + int(self.cfg.seed)) % 2)

        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        cx = float(rng.uniform(0.44, 0.56) * (size - 1))
        cy = float(rng.uniform(0.44, 0.56) * (size - 1))
        radius = float(rng.uniform(0.24, 0.30) * size)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        face = (dist <= radius).astype(np.float32)
        image = np.full((size, size), 0.08, dtype=np.float32)
        image += face * 0.58
        image += face * (0.10 * (1.0 - np.clip(dist / max(radius, 1e-6), 0.0, 1.0)))

        eye_dx = 0.34 * radius
        eye_y = cy - 0.14 * radius
        eye_sigma = 1.4
        for eye_x in (cx - eye_dx, cx + eye_dx):
            image -= 0.35 * np.exp(
                -((xx - eye_x) ** 2 + (yy - eye_y) ** 2) / (2.0 * eye_sigma * eye_sigma)
            ).astype(np.float32)

        image += 0.10 * np.exp(
            -((xx - cx) ** 2 + (yy - (cy + 0.06 * radius)) ** 2) / (2.0 * 1.6 * 1.6)
        ).astype(np.float32)

        mouth = np.exp(
            -((yy - (cy + 0.28 * radius)) ** 2) / (2.0 * 1.0 * 1.0)
            - ((xx - cx) ** 2) / (2.0 * (0.22 * radius) ** 2)
        ).astype(np.float32)
        image -= 0.12 * mouth

        if label == 0:
            # Spoof: add display border, stripe artifacts, and lower local contrast.
            border = np.zeros_like(image)
            border[[2, 3, size - 4, size - 3], :] = 1.0
            border[:, [2, 3, size - 4, size - 3]] = 1.0
            image += 0.18 * border
            stripes = (np.sin((xx + yy) * 0.55) > 0).astype(np.float32)
            image = 0.65 * image + 0.20 * stripes
            image = np.roll(image, shift=int(rng.integers(-1, 2)), axis=1)
        else:
            # Live: mild shading asymmetry and cleaner details.
            image += 0.08 * np.clip((xx - cx) / max(radius, 1.0), -1.0, 1.0) * face

        image += rng.normal(0.0, float(self.cfg.noise_std), size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        return torch.from_numpy(image).unsqueeze(0), label


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticFaceLivenessDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticFaceLivenessDataset", "get_dataloaders"]
