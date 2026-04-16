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
    num_templates: int = 8
    noise_std: float = 0.045


def _render_template(*, template_id: int, variation_seed: int, size: int, noise_std: float) -> np.ndarray:
    rng = np.random.default_rng(int(template_id) * 100_003 + int(variation_seed) * 97)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    image = np.full((size, size), 0.06, dtype=np.float32)

    row = int(template_id) // 4
    col = int(template_id) % 4
    image += 0.06 * (yy / max(size - 1, 1)).astype(np.float32)

    cx = float((0.18 + 0.18 * col + rng.uniform(-0.02, 0.02)) * (size - 1))
    cy = float((0.30 + 0.18 * row + rng.uniform(-0.03, 0.03)) * (size - 1))
    sigma_x = float((0.08 + 0.01 * col) * size)
    sigma_y = float((0.12 + 0.02 * row) * size)
    blob = np.exp(-((xx - cx) ** 2) / (2.0 * sigma_x**2) - ((yy - cy) ** 2) / (2.0 * sigma_y**2)).astype(
        np.float32
    )
    image += 0.32 * blob

    angle = float((template_id + 1) * np.pi / 10.0)
    centered_x = xx - size * 0.5
    centered_y = yy - size * 0.5
    stripe = np.exp(
        -((centered_x * np.cos(angle) + centered_y * np.sin(angle)) ** 2)
        / (2.0 * (1.0 + 0.1 * template_id) ** 2)
    ).astype(np.float32)
    image += 0.14 * stripe

    box_h = int(max(4, size * (0.10 + 0.01 * row)))
    box_w = int(max(4, size * (0.12 + 0.01 * col)))
    top = int(np.clip((0.58 - 0.06 * row) * size + rng.integers(-2, 3), 0, size - box_h))
    left = int(np.clip((0.18 + 0.16 * col) * size + rng.integers(-2, 3), 0, size - box_w))
    image[top : top + box_h, left : left + box_w] += 0.20 + 0.02 * template_id

    notch_y = int(np.clip((0.18 + 0.10 * row) * size, 0, size - 1))
    image[max(0, notch_y - 1) : min(size, notch_y + 2), :] -= 0.05 + 0.01 * col

    image = np.roll(image, shift=int(rng.integers(-2, 3)), axis=0)
    image = np.roll(image, shift=int(rng.integers(-2, 3)), axis=1)
    image += rng.normal(0.0, float(noise_std), size=(size, size)).astype(np.float32)
    return np.clip(image, 0.0, 1.0)


class SyntheticImageMatchingDataset:
    """Binary matching over synthetic template pairs."""

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if int(cfg.in_channels) != 1:
            raise ValueError("This lesson expects grayscale inputs.")
        if int(cfg.num_templates) < 2:
            raise ValueError("num_templates must be >= 2")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int):
        import torch

        seed = int(self.cfg.seed)
        template_count = int(self.cfg.num_templates)
        label = int((int(idx) + seed) % 2)
        template_a = int((int(idx) * 5 + seed * 11) % template_count)
        if label == 1:
            template_b = template_a
        else:
            template_b = (template_a + 1 + (int(idx) * 3 + seed) % (template_count - 1)) % template_count

        image_a = _render_template(
            template_id=template_a,
            variation_seed=seed * 10_000 + int(idx) * 2,
            size=int(self.cfg.image_size),
            noise_std=float(self.cfg.noise_std),
        )
        image_b = _render_template(
            template_id=template_b,
            variation_seed=seed * 10_000 + int(idx) * 2 + 1,
            size=int(self.cfg.image_size),
            noise_std=float(self.cfg.noise_std),
        )
        return torch.from_numpy(image_a).unsqueeze(0), torch.from_numpy(image_b).unsqueeze(0), label


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticImageMatchingDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch):
        images_a = torch.stack([item[0] for item in batch], dim=0)
        images_b = torch.stack([item[1] for item in batch], dim=0)
        labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
        return images_a, images_b, labels

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


__all__ = ["DataConfig", "SyntheticImageMatchingDataset", "get_dataloaders"]
