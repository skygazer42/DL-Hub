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


class SyntheticFingerSpreadDataset:
    """Synthetic grayscale hand crops with a normalized finger spread target."""

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
        spread = float(rng.uniform(0.02, 0.98))

        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        image = np.full((size, size), 0.05, dtype=np.float32)

        cx = float(0.5 * (size - 1) + rng.uniform(-2.0, 2.0))
        cy = float(0.62 * (size - 1) + rng.uniform(-2.0, 2.0))
        palm = np.exp(
            -((xx - cx) ** 2) / (2.0 * (0.17 * size) ** 2) - ((yy - cy) ** 2) / (2.0 * (0.13 * size) ** 2)
        ).astype(np.float32)
        image += 0.56 * palm

        finger_sep = (0.04 + 0.08 * spread) * size
        finger_width = 0.028 * size
        finger_height = (0.11 + 0.03 * spread) * size
        finger_y = cy - 0.23 * size
        for offset in (-1.5, -0.5, 0.5, 1.5):
            fx = cx + offset * finger_sep + rng.uniform(-0.6, 0.6)
            fy = finger_y + rng.uniform(-0.8, 0.8)
            finger = np.exp(
                -((xx - fx) ** 2) / (2.0 * finger_width**2) - ((yy - fy) ** 2) / (2.0 * finger_height**2)
            ).astype(np.float32)
            image += 0.18 * finger

        thumb_cx = cx - (0.18 + 0.04 * spread) * size
        thumb_cy = cy - 0.03 * size
        thumb = np.exp(
            -((xx - thumb_cx) ** 2) / (2.0 * (0.07 * size) ** 2)
            - ((yy - thumb_cy) ** 2) / (2.0 * (0.045 * size) ** 2)
        ).astype(np.float32)
        image += 0.14 * thumb

        image += 0.05 * np.clip((yy - cy) / max(0.35 * size, 1.0), -1.0, 1.0)
        image += rng.normal(0.0, float(self.cfg.noise_std), size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        target = np.array([spread], dtype=np.float32)
        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(target)


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticFingerSpreadDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticFingerSpreadDataset", "get_dataloaders"]
