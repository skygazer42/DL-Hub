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
    noise_std: float = 0.03


class SyntheticGazeDataset:
    """Synthetic grayscale face crops with normalized gaze x/y targets."""

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
        rng = np.random.default_rng(int(self.cfg.seed) * 1_000_003 + int(idx))
        gaze_x = float(rng.uniform(0.08, 0.92))
        gaze_y = float(rng.uniform(0.08, 0.92))

        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        image = np.full((size, size), 0.06, dtype=np.float32)

        cx = float(0.5 * (size - 1) + rng.uniform(-1.5, 1.5))
        cy = float(0.56 * (size - 1) + rng.uniform(-1.0, 1.0))
        rx = float(0.24 * size)
        ry = float(0.30 * size)
        face_mask = (((xx - cx) / max(rx, 1e-6)) ** 2 + ((yy - cy) / max(ry, 1e-6)) ** 2 <= 1.0).astype(
            np.float32
        )
        image += 0.62 * face_mask

        eye_y = cy - 0.18 * ry
        eye_dx = 0.40 * rx
        eye_rx = max(0.18 * rx, 2.0)
        eye_ry = max(0.10 * ry, 1.6)
        pupil_shift_x = (gaze_x - 0.5) * eye_rx * 1.4
        pupil_shift_y = (gaze_y - 0.5) * eye_ry * 1.4

        for eye_cx in (cx - eye_dx, cx + eye_dx):
            eye_mask = (((xx - eye_cx) / eye_rx) ** 2 + ((yy - eye_y) / eye_ry) ** 2 <= 1.0).astype(np.float32)
            image += 0.20 * eye_mask
            pupil = np.exp(
                -((xx - (eye_cx + pupil_shift_x)) ** 2) / (2.0 * max(0.20 * eye_rx, 1.0) ** 2)
                - ((yy - (eye_y + pupil_shift_y)) ** 2) / (2.0 * max(0.35 * eye_ry, 1.0) ** 2)
            ).astype(np.float32)
            image -= 0.42 * pupil

        nose = np.exp(
            -((xx - cx) ** 2) / (2.0 * max(0.12 * rx, 1.0) ** 2)
            - ((yy - (cy + 0.05 * ry)) ** 2) / (2.0 * max(0.18 * ry, 1.0) ** 2)
        ).astype(np.float32)
        image -= 0.08 * nose

        mouth = np.exp(
            -((yy - (cy + 0.34 * ry)) ** 2) / (2.0 * 1.2**2)
            - ((xx - cx) ** 2) / (2.0 * max(0.30 * rx, 1.5) ** 2)
        ).astype(np.float32)
        image -= 0.10 * mouth

        image += 0.05 * np.clip((xx - cx) / max(rx, 1.0), -1.0, 1.0) * face_mask
        image += rng.normal(0.0, float(self.cfg.noise_std), size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        target = np.array([gaze_x, gaze_y], dtype=np.float32)
        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(target)


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticGazeDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticGazeDataset", "get_dataloaders"]
