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
    noise_std: float = 0.035


class SyntheticFacePoseDataset:
    """Synthetic face crops with normalized yaw, pitch, and roll targets."""

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
        rng = np.random.default_rng(int(self.cfg.seed) * 2_000_071 + int(idx))
        yaw = float(rng.uniform(-1.0, 1.0))
        pitch = float(rng.uniform(-1.0, 1.0))
        roll = float(rng.uniform(-1.0, 1.0))

        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        cx = float(0.5 * (size - 1) + yaw * 2.4)
        cy = float(0.55 * (size - 1) + pitch * 1.8)
        rx = float(0.23 * size)
        ry = float(0.28 * size)

        face_mask = (((xx - cx) / max(rx, 1e-6)) ** 2 + ((yy - cy) / max(ry, 1e-6)) ** 2 <= 1.0).astype(
            np.float32
        )
        image = np.full((size, size), 0.08, dtype=np.float32)
        image += 0.60 * face_mask

        angle = float(roll * 0.45)
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))

        def _rotate(offset_x: float, offset_y: float) -> tuple[float, float]:
            return (
                cos_a * offset_x - sin_a * offset_y,
                sin_a * offset_x + cos_a * offset_y,
            )

        eye_dx = 0.32 * rx * (1.0 - 0.18 * yaw)
        eye_y = -0.18 * ry - 0.08 * pitch * ry
        left_eye = _rotate(-eye_dx, eye_y)
        right_eye = _rotate(eye_dx, eye_y)
        eye_sigma = 1.4
        for off_x, off_y in (left_eye, right_eye):
            eye = np.exp(-((xx - (cx + off_x)) ** 2 + (yy - (cy + off_y)) ** 2) / (2.0 * eye_sigma * eye_sigma))
            image -= 0.30 * eye.astype(np.float32)

        brow_y = -0.27 * ry - 0.10 * pitch * ry
        brow_off = _rotate(0.0, brow_y)[1]
        brow_curve = np.exp(-((yy - (cy + brow_off)) ** 2) / (2.0 * 0.8 * 0.8)).astype(np.float32)
        image -= 0.05 * brow_curve * np.clip(1.0 - np.abs(xx - cx) / max(0.45 * rx, 1.0), 0.0, 1.0)

        nose_center = _rotate(0.10 * yaw * rx, 0.02 * pitch * ry)
        nose = np.exp(
            -((xx - (cx + nose_center[0])) ** 2) / (2.0 * max(0.10 * rx, 1.0) ** 2)
            - ((yy - (cy + nose_center[1])) ** 2) / (2.0 * max(0.18 * ry, 1.0) ** 2)
        ).astype(np.float32)
        image -= 0.09 * nose

        mouth_offset = _rotate(0.12 * yaw * rx, 0.30 * ry + 0.10 * pitch * ry)
        mouth_width = max((0.24 + 0.05 * abs(yaw)) * rx, 1.0)
        mouth = np.exp(
            -((yy - (cy + mouth_offset[1])) ** 2) / (2.0 * 1.2 * 1.2)
            - ((xx - (cx + mouth_offset[0])) ** 2) / (2.0 * mouth_width**2)
        ).astype(np.float32)
        image -= 0.13 * mouth

        image += 0.05 * yaw * np.clip((xx - cx) / max(rx, 1.0), -1.0, 1.0) * face_mask
        image += rng.normal(0.0, float(self.cfg.noise_std), size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        pose = np.array([yaw, pitch, roll], dtype=np.float32)
        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(pose)


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticFacePoseDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )

    def _collate(batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        poses = torch.stack([item[1] for item in batch], dim=0).to(torch.float32)
        return images, poses

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


__all__ = ["DataConfig", "SyntheticFacePoseDataset", "get_dataloaders"]
