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
    num_classes: int = 6
    noise_std: float = 0.035


def _render_species(*, label: int, variation_seed: int, size: int, noise_std: float) -> np.ndarray:
    rng = np.random.default_rng(int(label) * 100_003 + int(variation_seed) * 53)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    image = np.full((size, size), 0.07, dtype=np.float32)

    cx = float((0.50 + rng.uniform(-0.03, 0.03)) * (size - 1))
    cy = float((0.52 + rng.uniform(-0.03, 0.03)) * (size - 1))
    rx = float((0.20 + 0.01 * (label % 3)) * size)
    ry = float((0.30 + 0.01 * (label // 3)) * size)
    dist = ((xx - cx) / max(rx, 1e-6)) ** 2 + ((yy - cy) / max(ry, 1e-6)) ** 2
    silhouette = (dist <= 1.0).astype(np.float32)
    image += 0.46 * silhouette

    vein_angle = float((-0.28 + 0.10 * label) * np.pi)
    axis = (xx - cx) * np.cos(vein_angle) + (yy - cy) * np.sin(vein_angle)
    primary_vein = np.exp(-(axis**2) / (2.0 * (1.2 + 0.1 * (label % 2)) ** 2)).astype(np.float32)
    image -= (0.12 + 0.01 * (label % 3)) * primary_vein * silhouette

    stripe_freq = 0.16 + 0.02 * label
    stripe_phase = 0.5 * label
    stripes = (0.5 + 0.5 * np.cos(stripe_freq * (xx - cx) + stripe_phase)).astype(np.float32)
    image += (0.08 + 0.01 * (label // 2)) * stripes * silhouette

    notch_offset = (-0.18, -0.10, -0.02, 0.06, 0.14, 0.22)[label]
    notch = np.exp(
        -((xx - (cx + notch_offset * rx)) ** 2) / (2.0 * 2.0**2)
        - ((yy - (cy + 0.42 * ry)) ** 2) / (2.0 * 1.6**2)
    ).astype(np.float32)
    image -= (0.16 + 0.01 * label) * notch

    tip_shine = np.exp(
        -((xx - cx) ** 2) / (2.0 * (0.12 * size) ** 2) - ((yy - (cy - 0.52 * ry)) ** 2) / (2.0 * 1.8**2)
    ).astype(np.float32)
    image += (0.07 + 0.01 * (label % 2)) * tip_shine

    image += rng.normal(0.0, float(noise_std), size=(size, size)).astype(np.float32)
    return np.clip(image, 0.0, 1.0)


class SyntheticFineGrainedRecognitionDataset:
    """Synthetic fine-grained classification with small inter-class differences."""

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if int(cfg.in_channels) != 1:
            raise ValueError("This lesson expects grayscale inputs.")
        if int(cfg.num_classes) != 6:
            raise ValueError("This lesson uses exactly 6 fine-grained classes.")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int):
        import torch

        label = int((int(idx) + int(self.cfg.seed)) % int(self.cfg.num_classes))
        image = _render_species(
            label=label,
            variation_seed=int(self.cfg.seed) * 10_000 + int(idx),
            size=int(self.cfg.image_size),
            noise_std=float(self.cfg.noise_std),
        )
        return torch.from_numpy(image).unsqueeze(0), label


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticFineGrainedRecognitionDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
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


__all__ = ["DataConfig", "SyntheticFineGrainedRecognitionDataset", "get_dataloaders"]
