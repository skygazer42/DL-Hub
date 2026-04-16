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
    num_attributes: int = 3
    noise_std: float = 0.04


class SyntheticFaceAttributeDataset:
    """Synthetic face crops with 3 binary attributes: smile, glasses, beard."""

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if int(cfg.num_attributes) != 3:
            raise ValueError("This lesson uses exactly 3 attributes.")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int):
        import torch

        size = int(self.cfg.image_size)
        rng = np.random.default_rng(int(self.cfg.seed) * 2_000_033 + int(idx))
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

        cx = float(rng.uniform(0.35, 0.65) * (size - 1))
        cy = float(rng.uniform(0.36, 0.64) * (size - 1))
        rx = float(rng.uniform(0.18, 0.27) * size)
        ry = float(rng.uniform(0.22, 0.30) * size)

        smiling = float(rng.random() < 0.5)
        glasses = float(rng.random() < 0.35)
        beard = float(rng.random() < 0.4)

        face_mask = (((xx - cx) / max(rx, 1e-6)) ** 2 + ((yy - cy) / max(ry, 1e-6)) ** 2) <= 1.0
        face_mask = face_mask.astype(np.float32)

        image = np.full((size, size), 0.08, dtype=np.float32)
        image += 0.62 * face_mask

        eye_y = cy - 0.18 * ry
        eye_dx = 0.34 * rx
        for eye_x in (cx - eye_dx, cx + eye_dx):
            eye = np.exp(-((xx - eye_x) ** 2 + (yy - eye_y) ** 2) / (2.0 * 1.3 * 1.3))
            image -= 0.34 * eye.astype(np.float32)

        if glasses > 0.5:
            for eye_x in (cx - eye_dx, cx + eye_dx):
                frame = (
                    (((xx - eye_x) / max(0.18 * rx, 1.0)) ** 2 + ((yy - eye_y) / max(0.16 * ry, 1.0)) ** 2)
                    <= 1.0
                )
                image -= 0.10 * frame.astype(np.float32)
            bridge = np.exp(
                -((yy - eye_y) ** 2) / (2.0 * 0.7 * 0.7)
                - ((xx - cx) ** 2) / (2.0 * (0.12 * rx) ** 2)
            )
            image -= 0.08 * bridge.astype(np.float32)

        mouth_center_y = cy + 0.30 * ry
        mouth = np.exp(
            -((yy - mouth_center_y) ** 2) / (2.0 * 1.0 * 1.0)
            - ((xx - cx) ** 2) / (2.0 * (0.28 * rx) ** 2)
        )
        if smiling > 0.5:
            image -= 0.22 * mouth.astype(np.float32)
        else:
            image -= 0.10 * mouth.astype(np.float32)
            image += 0.04 * np.exp(-((yy - (mouth_center_y + 1.3)) ** 2) / (2.0 * 1.0 * 1.0)).astype(
                np.float32
            )

        if beard > 0.5:
            beard_region = ((yy > cy + 0.18 * ry) & (face_mask > 0.0)).astype(np.float32)
            coarse_texture = (np.sin(0.40 * xx + 0.25 * yy) + np.cos(0.22 * xx - 0.17 * yy)) * 0.5
            image -= beard_region * (0.11 + 0.05 * coarse_texture.astype(np.float32))

        image += 0.08 * np.clip((xx - cx) / max(rx, 1.0), -1.0, 1.0) * face_mask
        image += rng.normal(0.0, float(self.cfg.noise_std), size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        target = np.asarray([smiling, glasses, beard], dtype=np.float32)
        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(target)


def get_dataloaders(cfg: DataConfig):
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticFaceAttributeDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticFaceAttributeDataset", "get_dataloaders"]
