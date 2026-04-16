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
    num_landmarks: int = 5
    noise_std: float = 0.04


class SyntheticFaceLandmarkDataset:
    """Simple synthetic faces with five normalized landmarks."""

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if int(cfg.num_landmarks) != 5:
            raise ValueError("This lesson uses exactly 5 landmarks.")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _sample_geometry(self, idx: int) -> tuple[np.ndarray, float, float, float]:
        rng = np.random.default_rng(int(self.cfg.seed) * 1_000_003 + int(idx))
        size = int(self.cfg.image_size)
        cx = float(rng.uniform(0.42, 0.58) * (size - 1))
        cy = float(rng.uniform(0.42, 0.58) * (size - 1))
        radius = float(rng.uniform(0.24, 0.31) * size)

        eye_dx = float(rng.uniform(0.32, 0.40) * radius)
        eye_y = cy - float(rng.uniform(0.10, 0.18) * radius)
        mouth_y = cy + float(rng.uniform(0.24, 0.34) * radius)
        mouth_dx = float(rng.uniform(0.22, 0.32) * radius)
        nose_y = cy + float(rng.uniform(0.02, 0.10) * radius)

        landmarks = np.array(
            [
                [cx - eye_dx, eye_y],
                [cx + eye_dx, eye_y],
                [cx, nose_y],
                [cx - mouth_dx, mouth_y],
                [cx + mouth_dx, mouth_y],
            ],
            dtype=np.float32,
        )
        return landmarks, cx, cy, radius

    def __getitem__(self, idx: int):
        import torch

        size = int(self.cfg.image_size)
        landmarks, cx, cy, radius = self._sample_geometry(int(idx))
        rng = np.random.default_rng(int(self.cfg.seed) * 97_409 + int(idx))

        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        face_mask = (dist <= radius).astype(np.float32)

        image = np.full((size, size), 0.08, dtype=np.float32)
        image += face_mask * 0.58
        image += face_mask * (0.08 * (1.0 - np.clip(dist / max(radius, 1e-6), 0.0, 1.0)))

        for point_idx, (x, y) in enumerate(landmarks):
            sigma = 1.2 if point_idx != 2 else 1.4
            blob = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
            if point_idx in (0, 1):
                image -= 0.38 * blob
            elif point_idx == 2:
                image += 0.12 * blob
            else:
                image -= 0.16 * blob

        mouth_center_y = landmarks[3:5, 1].mean()
        mouth_center_x = landmarks[3:5, 0].mean()
        mouth_curve = np.exp(
            -((yy - mouth_center_y) ** 2) / (2.0 * 1.0 * 1.0)
            - ((xx - mouth_center_x) ** 2) / (2.0 * (0.25 * radius) ** 2)
        ).astype(np.float32)
        image -= 0.08 * mouth_curve

        image += rng.normal(0.0, float(self.cfg.noise_std), size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        landmark_tensor = torch.from_numpy((landmarks / float(size - 1)).reshape(-1))
        return image_tensor, landmark_tensor.to(torch.float32)


def get_dataloaders(cfg: DataConfig):
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticFaceLandmarkDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticFaceLandmarkDataset", "get_dataloaders"]
