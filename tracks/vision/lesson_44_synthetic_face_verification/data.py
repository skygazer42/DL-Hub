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
    num_identities: int = 24
    noise_std: float = 0.05


def _render_face(*, identity: int, variation_seed: int, size: int, noise_std: float) -> np.ndarray:
    identity_rng = np.random.default_rng(int(identity) * 1_000_003 + 17)
    var_rng = np.random.default_rng(int(variation_seed) * 1_000_003 + 29)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = float((0.36 + 0.28 * identity_rng.random()) * (size - 1))
    cy = float((0.38 + 0.24 * identity_rng.random()) * (size - 1))
    radius = float((0.23 + 0.09 * identity_rng.random()) * size)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    face = (dist <= radius).astype(np.float32)
    image = np.full((size, size), 0.08, dtype=np.float32)
    image += face * 0.54
    image += face * (0.15 * (1.0 - np.clip(dist / max(radius, 1e-6), 0.0, 1.0)))

    eye_dx = float((0.28 + 0.08 * identity_rng.random()) * radius)
    eye_y = float(cy - (0.10 + 0.06 * identity_rng.random()) * radius)
    eye_sigma = float(1.2 + 0.8 * identity_rng.random())
    for eye_x in (cx - eye_dx, cx + eye_dx):
        image -= 0.35 * np.exp(
            -((xx - eye_x) ** 2 + (yy - eye_y) ** 2) / (2.0 * eye_sigma * eye_sigma)
        ).astype(np.float32)

    mouth_y = float(cy + (0.24 + 0.08 * identity_rng.random()) * radius)
    mouth_w = float((0.18 + 0.10 * identity_rng.random()) * radius)
    mouth_h = float(0.8 + 0.8 * identity_rng.random())
    mouth = np.exp(
        -((yy - mouth_y) ** 2) / (2.0 * mouth_h * mouth_h) - ((xx - cx) ** 2) / (2.0 * mouth_w * mouth_w)
    ).astype(np.float32)
    image -= 0.14 * mouth

    nose = np.exp(
        -((xx - cx) ** 2) / (2.0 * 1.8 * 1.8) - ((yy - (cy + 0.06 * radius)) ** 2) / (2.0 * 3.2 * 3.2)
    ).astype(np.float32)
    image += 0.08 * nose

    # Per-sample capture variation for same-identity pairs.
    image = np.roll(image, shift=int(var_rng.integers(-2, 3)), axis=1)
    image = np.roll(image, shift=int(var_rng.integers(-2, 3)), axis=0)
    image += (var_rng.uniform(-0.08, 0.08) * np.clip((xx - cx) / max(radius, 1.0), -1.0, 1.0) * face).astype(
        np.float32
    )
    image += (var_rng.uniform(-0.06, 0.06) * np.clip((cy - yy) / max(radius, 1.0), -1.0, 1.0) * face).astype(
        np.float32
    )
    image += var_rng.normal(0.0, float(noise_std), size=(size, size)).astype(np.float32)
    return np.clip(image, 0.0, 1.0)


class SyntheticFaceVerificationDataset:
    """Binary same-identity verification with synthetic paired faces."""

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if int(cfg.num_identities) < 2:
            raise ValueError("num_identities must be >= 2")
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int):
        import torch

        seed = int(self.cfg.seed)
        n_ids = int(self.cfg.num_identities)
        size = int(self.cfg.image_size)
        label = int((int(idx) + seed) % 2)

        base_identity = int((int(idx) * 17 + seed * 13) % n_ids)
        if label == 1:
            identity_a = base_identity
            identity_b = base_identity
        else:
            identity_a = base_identity
            identity_b = (base_identity + 1 + (int(idx) * 7 + seed) % (n_ids - 1)) % n_ids

        image_a = _render_face(
            identity=identity_a,
            variation_seed=seed * 10_000 + int(idx) * 2,
            size=size,
            noise_std=float(self.cfg.noise_std),
        )
        image_b = _render_face(
            identity=identity_b,
            variation_seed=seed * 10_000 + int(idx) * 2 + 1,
            size=size,
            noise_std=float(self.cfg.noise_std),
        )
        return torch.from_numpy(image_a).unsqueeze(0), torch.from_numpy(image_b).unsqueeze(0), label


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticFaceVerificationDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticFaceVerificationDataset", "get_dataloaders"]
