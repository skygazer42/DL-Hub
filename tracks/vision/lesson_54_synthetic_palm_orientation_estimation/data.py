from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 2048
    batch_size: int = 32
    image_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    in_channels: int = 1
    noise_std: float = 0.03


class SyntheticPalmOrientationDataset:
    """Synthetic grayscale palm crops with a normalized orientation target."""

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 48:
            raise ValueError("image_size must be >= 48")
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
        rng = np.random.default_rng(int(self.cfg.seed) * 1_000_003 + int(idx))
        orientation = float(rng.uniform(0.05, 0.95))
        angle = (orientation - 0.5) * np.deg2rad(140.0)

        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        image = np.full((size, size), 0.05, dtype=np.float32)

        cx = float(0.5 * (size - 1) + rng.uniform(-2.0, 2.0))
        cy = float(0.56 * (size - 1) + rng.uniform(-2.0, 2.0))
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))

        x_shift = xx - cx
        y_shift = yy - cy
        long_axis = x_shift * cos_a + y_shift * sin_a
        short_axis = -x_shift * sin_a + y_shift * cos_a

        palm = np.exp(
            -(long_axis**2) / (2.0 * (0.26 * size) ** 2) - (short_axis**2) / (2.0 * (0.12 * size) ** 2)
        ).astype(np.float32)
        image += 0.62 * palm

        thumb_cx = cx + 0.10 * size * cos_a - 0.18 * size * sin_a
        thumb_cy = cy + 0.10 * size * sin_a + 0.18 * size * cos_a
        thumb = np.exp(
            -((xx - thumb_cx) ** 2) / (2.0 * (0.08 * size) ** 2)
            - ((yy - thumb_cy) ** 2) / (2.0 * (0.05 * size) ** 2)
        ).astype(np.float32)
        image += 0.18 * thumb

        finger_base_x = cx + 0.20 * size * cos_a
        finger_base_y = cy + 0.20 * size * sin_a
        for offset in (-0.16, -0.05, 0.05, 0.16):
            fx = finger_base_x - offset * size * sin_a
            fy = finger_base_y + offset * size * cos_a
            finger = np.exp(
                -((xx - fx) ** 2) / (2.0 * (0.04 * size) ** 2)
                - ((yy - fy) ** 2) / (2.0 * (0.09 * size) ** 2)
            ).astype(np.float32)
            image += 0.12 * finger

        image += 0.05 * np.clip(long_axis / max(0.30 * size, 1.0), -1.0, 1.0)
        image += rng.normal(0.0, float(self.cfg.noise_std), size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        target = np.array([orientation], dtype=np.float32)
        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(target)


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticPalmOrientationDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )

    def _collate(batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        targets = torch.stack([item[1] for item in batch], dim=0).to(torch.float32)
        return images, targets

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


__all__ = ["DataConfig", "SyntheticPalmOrientationDataset", "get_dataloaders"]
