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
    num_identities: int = 5
    noise_std: float = 0.035


class SyntheticFaceRetrievalDataset:
    """Synthetic face crops that expose identity labels for metric learning."""

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if int(cfg.in_channels) != 1:
            raise ValueError("This lesson expects grayscale inputs.")
        if int(cfg.num_identities) != 5:
            raise ValueError("This lesson uses exactly 5 identity classes.")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int):
        import torch

        size = int(self.cfg.image_size)
        rng = np.random.default_rng(int(self.cfg.seed) * 2_000_071 + int(idx))
        identity = int((int(idx) + int(self.cfg.seed)) % int(self.cfg.num_identities))

        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        cx = float(rng.uniform(0.40, 0.60) * (size - 1))
        cy = float(rng.uniform(0.42, 0.62) * (size - 1))
        rx = float(rng.uniform(0.18, 0.27) * size)
        ry = float(rng.uniform(0.22, 0.30) * size)

        face_mask = (((xx - cx) / max(rx, 1e-6)) ** 2 + ((yy - cy) / max(ry, 1e-6)) ** 2 <= 1.0).astype(
            np.float32
        )
        image = np.full((size, size), 0.08, dtype=np.float32)
        image += 0.60 * face_mask

        eye_dx_scale = (0.26, 0.34, 0.30, 0.22, 0.38)
        eye_y_scale = (0.18, 0.15, 0.20, 0.17, 0.13)
        mouth_width_scale = (0.22, 0.34, 0.28, 0.18, 0.30)
        mouth_height_shift = (-0.8, 0.6, 0.0, -1.2, 1.0)
        brow_strength = (0.04, 0.09, 0.06, 0.03, 0.08)
        nose_width_scale = (0.10, 0.15, 0.12, 0.08, 0.18)
        side_shading = (-0.06, 0.08, -0.02, 0.05, -0.08)

        eye_y = cy - eye_y_scale[identity] * ry
        eye_dx = eye_dx_scale[identity] * rx
        eye_sigma = 1.2 + 0.15 * identity
        for eye_x in (cx - eye_dx, cx + eye_dx):
            eye = np.exp(-((xx - eye_x) ** 2 + (yy - eye_y) ** 2) / (2.0 * eye_sigma * eye_sigma))
            image -= (0.28 + 0.01 * identity) * eye.astype(np.float32)

        brow_y = eye_y - 2.0
        brow_band = np.exp(-((yy - brow_y) ** 2) / (2.0 * 0.8 * 0.8)).astype(np.float32)
        brow_window = np.clip(1.0 - np.abs(xx - cx) / max(0.48 * rx, 1.0), 0.0, 1.0)
        image -= brow_strength[identity] * brow_band * brow_window

        nose_sigma_x = max(nose_width_scale[identity] * rx, 1.0)
        nose_sigma_y = max(0.16 * ry, 1.0)
        nose = np.exp(-((xx - cx) ** 2) / (2.0 * nose_sigma_x**2) - ((yy - cy) ** 2) / (2.0 * nose_sigma_y**2))
        image -= 0.09 * nose.astype(np.float32)

        mouth_center_y = cy + 0.29 * ry + mouth_height_shift[identity]
        mouth_width = max(mouth_width_scale[identity] * rx, 1.0)
        mouth = np.exp(
            -((yy - mouth_center_y) ** 2) / (2.0 * 1.0 * 1.0) - ((xx - cx) ** 2) / (2.0 * mouth_width**2)
        ).astype(np.float32)
        image -= (0.11 + 0.02 * identity) * mouth

        hairline = cy - (0.68 - 0.03 * identity) * ry
        image -= (0.10 + 0.01 * identity) * ((yy < hairline) * face_mask).astype(np.float32)

        image += side_shading[identity] * np.clip((xx - cx) / max(rx, 1.0), -1.0, 1.0) * face_mask
        image += rng.normal(0.0, float(self.cfg.noise_std), size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        return torch.from_numpy(image).unsqueeze(0), identity


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticFaceRetrievalDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticFaceRetrievalDataset", "get_dataloaders"]
