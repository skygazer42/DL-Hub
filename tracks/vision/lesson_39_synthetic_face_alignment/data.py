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
    max_rotation_deg: float = 18.0
    scale_jitter: float = 0.12
    translation_jitter: float = 0.08
    noise_std: float = 0.04


def canonical_landmarks() -> np.ndarray:
    return np.array(
        [
            [0.34, 0.37],
            [0.66, 0.37],
            [0.50, 0.53],
            [0.39, 0.69],
            [0.61, 0.69],
        ],
        dtype=np.float32,
    )


class SyntheticFaceAlignmentDataset:
    """Synthetic faces paired with a canonical five-point target layout."""

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if int(cfg.num_landmarks) != 5:
            raise ValueError("This lesson uses exactly 5 landmarks.")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        self.cfg = cfg
        self._canonical = canonical_landmarks()

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _sample_geometry(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(int(self.cfg.seed) * 1_000_003 + int(idx))
        base = self._canonical.copy()

        angle_deg = float(rng.uniform(-float(self.cfg.max_rotation_deg), float(self.cfg.max_rotation_deg)))
        angle = np.deg2rad(angle_deg)
        scale = float(rng.uniform(1.0 - float(self.cfg.scale_jitter), 1.0 + float(self.cfg.scale_jitter)))
        dx = float(rng.uniform(-float(self.cfg.translation_jitter), float(self.cfg.translation_jitter)))
        dy = float(rng.uniform(-float(self.cfg.translation_jitter), float(self.cfg.translation_jitter)))

        centered = base - 0.5
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )
        posed = centered @ rotation.T
        posed = posed * scale
        posed = posed + np.array([0.5 + dx, 0.5 + dy], dtype=np.float32)
        posed = np.clip(posed, 0.08, 0.92)
        return posed.astype(np.float32), base.astype(np.float32)

    def __getitem__(self, idx: int):
        import torch

        size = int(self.cfg.image_size)
        posed_landmarks, canonical = self._sample_geometry(int(idx))
        rng = np.random.default_rng(int(self.cfg.seed) * 97_409 + int(idx))

        posed_pixels = posed_landmarks * float(size - 1)
        eye_center = posed_pixels[:2].mean(axis=0)
        mouth_center = posed_pixels[3:].mean(axis=0)
        face_center = posed_pixels.mean(axis=0)
        radius = max(float(np.linalg.norm(eye_center - mouth_center) * 0.95), 0.22 * size)

        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        face_mask = (((xx - face_center[0]) / (0.9 * radius)) ** 2 + ((yy - face_center[1]) / radius) ** 2 <= 1.0)
        face_mask = face_mask.astype(np.float32)

        dist = np.sqrt((xx - face_center[0]) ** 2 + (yy - face_center[1]) ** 2)
        image = np.full((size, size), 0.08, dtype=np.float32)
        image += face_mask * 0.58
        image += face_mask * (0.08 * (1.0 - np.clip(dist / max(radius, 1e-6), 0.0, 1.0)))

        for point_idx, (x, y) in enumerate(posed_pixels):
            sigma = 1.2 if point_idx != 2 else 1.4
            blob = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
            if point_idx in (0, 1):
                image -= 0.38 * blob
            elif point_idx == 2:
                image += 0.12 * blob
            else:
                image -= 0.16 * blob

        mouth_center_y = posed_pixels[3:5, 1].mean()
        mouth_center_x = posed_pixels[3:5, 0].mean()
        mouth_curve = np.exp(
            -((yy - mouth_center_y) ** 2) / (2.0 * 1.0 * 1.0)
            - ((xx - mouth_center_x) ** 2) / (2.0 * (0.25 * radius) ** 2)
        ).astype(np.float32)
        image -= 0.08 * mouth_curve

        image += rng.normal(0.0, float(self.cfg.noise_std), size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        aligned_tensor = torch.from_numpy(canonical.reshape(-1)).to(torch.float32)
        return image_tensor, aligned_tensor


def get_dataloaders(cfg: DataConfig):
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticFaceAlignmentDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
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


__all__ = ["DataConfig", "SyntheticFaceAlignmentDataset", "canonical_landmarks", "get_dataloaders"]
