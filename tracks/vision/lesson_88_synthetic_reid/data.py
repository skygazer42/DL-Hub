from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 512
    batch_size: int = 16
    image_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    in_channels: int = 3
    num_identities: int = 8
    noise_std: float = 0.02


def _identity_palette(identity: int, num_identities: int) -> np.ndarray:
    phase = 2.0 * np.pi * (float(identity) / max(1.0, float(num_identities)))
    rgb = np.asarray(
        [
            0.45 + 0.35 * np.sin(phase),
            0.45 + 0.35 * np.sin(phase + 2.1),
            0.45 + 0.35 * np.sin(phase + 4.2),
        ],
        dtype=np.float32,
    )
    return np.clip(rgb, 0.08, 0.92)


def _render_person_like(
    *,
    identity: int,
    image_size: int,
    num_identities: int,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    h = int(image_size)
    w = int(image_size)
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h, dtype=np.float32),
        np.linspace(-1.0, 1.0, w, dtype=np.float32),
        indexing="ij",
    )

    # Camera variation.
    shift_x = float(rng.uniform(-0.18, 0.18))
    shift_y = float(rng.uniform(-0.12, 0.12))
    scale = float(rng.uniform(0.78, 1.05))

    x = (xx - shift_x) / scale
    y = (yy - shift_y) / scale

    # Identity-specific body style.
    body_w = 0.35 + 0.06 * np.sin(0.7 * identity)
    body_h = 0.58 + 0.05 * np.cos(0.5 * identity)
    head_r = 0.18 + 0.02 * np.sin(0.9 * identity)

    body = np.exp(-((x / body_w) ** 2 + ((y + 0.1) / body_h) ** 2) * 2.2)
    head = np.exp(-(((x) ** 2 + ((y - 0.52) ** 2)) / (head_r**2)) * 1.8)
    silhouette = np.clip(np.maximum(body, head), 0.0, 1.0)

    # Identity-coded stripe angle and frequency.
    angle = float((identity % max(2, num_identities)) / max(1, num_identities) * np.pi)
    freq = 3.0 + float(identity % 4)
    coord = np.cos(angle) * x + np.sin(angle) * (y + 0.1)
    stripes = 0.5 + 0.5 * np.sin(2.0 * np.pi * freq * coord)
    stripe_mask = (0.25 + 0.75 * stripes) * silhouette

    bg = np.stack(
        [
            0.12 + 0.06 * (1.0 - (yy + 1.0) * 0.5),
            0.12 + 0.04 * (xx + 1.0) * 0.5,
            0.14 + 0.05 * (1.0 - (xx + 1.0) * 0.5),
        ],
        axis=0,
    ).astype(np.float32)
    fg_color = _identity_palette(identity, num_identities).reshape(3, 1, 1)
    fg = fg_color * stripe_mask[None, :, :]

    image = bg * (1.0 - silhouette[None, :, :]) + fg
    image += rng.normal(0.0, float(noise_std), size=image.shape).astype(np.float32)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


class SyntheticReIDDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if int(cfg.in_channels) != 3:
            raise ValueError("in_channels must be 3 for this lesson")
        if int(cfg.num_identities) < 2:
            raise ValueError("num_identities must be >= 2")
        if int(cfg.num_samples) < int(cfg.num_identities):
            raise ValueError("num_samples must be >= num_identities")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if float(cfg.noise_std) < 0.0:
            raise ValueError("noise_std must be >= 0")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample_idx = int(idx)
        label = sample_idx % int(self.cfg.num_identities)
        seed = int(self.cfg.seed) * 1_000_003 + sample_idx
        image = _render_person_like(
            identity=label,
            image_size=int(self.cfg.image_size),
            num_identities=int(self.cfg.num_identities),
            noise_std=float(self.cfg.noise_std),
            seed=seed,
        )
        return torch.from_numpy(image), int(label)


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset = SyntheticReIDDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "SyntheticReIDDataset", "get_dataloaders"]

