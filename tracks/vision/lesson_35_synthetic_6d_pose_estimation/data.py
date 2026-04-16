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
    pose_dim: int = 9
    noise_std: float = 0.03


def _rotation_from_euler(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    return rz @ ry @ rx


class SyntheticPoseDataset:
    """Synthetic object views with a compact 6D rotation + translation target."""

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if int(cfg.pose_dim) != 9:
            raise ValueError("pose_dim must be 9 (6D rotation + 3D translation)")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int):
        import torch

        size = int(self.cfg.image_size)
        rng = np.random.default_rng(int(self.cfg.seed) * 1_700_003 + int(idx))
        yaw = float(rng.uniform(-0.9, 0.9))
        pitch = float(rng.uniform(-0.6, 0.6))
        roll = float(rng.uniform(-0.6, 0.6))
        tx = float(rng.uniform(-0.7, 0.7))
        ty = float(rng.uniform(-0.7, 0.7))
        tz = float(rng.uniform(-0.7, 0.7))

        rotation = _rotation_from_euler(yaw, pitch, roll)
        pose = np.concatenate([rotation[:, 0], rotation[:, 1], np.array([tx, ty, tz], dtype=np.float32)]).astype(
            np.float32
        )

        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        cx = (size - 1) * (0.5 + 0.22 * tx)
        cy = (size - 1) * (0.5 + 0.22 * ty)
        scale = 0.18 * size * (1.05 - 0.25 * tz)
        length = max(4.0, scale * (1.6 + 0.2 * np.cos(pitch)))
        width = max(2.0, scale * (0.55 + 0.2 * np.cos(roll)))

        rel_x = xx - cx
        rel_y = yy - cy
        u = np.cos(yaw) * rel_x + np.sin(yaw) * rel_y
        v = -np.sin(yaw) * rel_x + np.cos(yaw) * rel_y
        body_mask = (np.abs(u) <= length) & (np.abs(v) <= width)

        image = np.full((size, size), 0.08, dtype=np.float32)
        shading = 0.56 + 0.18 * pitch * (u / max(length, 1.0)) + 0.14 * roll * (v / max(width, 1.0))
        image += body_mask.astype(np.float32) * np.clip(shading, 0.18, 0.92)

        nose_x = cx + np.cos(yaw) * length * 0.85
        nose_y = cy + np.sin(yaw) * length * 0.85
        tail_x = cx - np.cos(yaw) * length * 0.85
        tail_y = cy - np.sin(yaw) * length * 0.85
        sigma = max(1.0, 0.10 * size)
        nose_blob = np.exp(-((xx - nose_x) ** 2 + (yy - nose_y) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
        tail_blob = np.exp(-((xx - tail_x) ** 2 + (yy - tail_y) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
        image += (0.18 + 0.08 * max(pitch, 0.0)) * nose_blob
        image -= (0.10 + 0.06 * max(-roll, 0.0)) * tail_blob

        image += rng.normal(0.0, float(self.cfg.noise_std), size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(pose)


def get_dataloaders(cfg: DataConfig):
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticPoseDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
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


__all__ = ["DataConfig", "SyntheticPoseDataset", "get_dataloaders"]
