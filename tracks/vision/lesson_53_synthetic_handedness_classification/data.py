from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 2048
    batch_size: int = 32
    image_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    in_channels: int = 1
    num_classes: int = 2
    noise_std: float = 0.02


class SyntheticHandednessDataset:
    """Render a compact synthetic hand crop and classify left vs right handedness.

    Label 0: left hand (thumb on the left side)
    Label 1: right hand (thumb on the right side)
    """

    class_names = ("left", "right")

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 48:
            raise ValueError("image_size must be >= 48")
        if int(cfg.in_channels) != 1:
            raise ValueError("This lesson expects grayscale inputs.")
        if int(cfg.num_classes) != 2:
            raise ValueError("This lesson uses exactly 2 classes.")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _sample_params(self, idx: int) -> tuple[int, dict[str, float]]:
        rng = np.random.default_rng(int(self.cfg.seed) * 1_000_003 + int(idx))
        label = int((int(idx) + int(self.cfg.seed)) % 2)

        size = int(self.cfg.image_size)
        center_x = float(rng.uniform(0.46, 0.54) * (size - 1))
        center_y = float(rng.uniform(0.52, 0.62) * (size - 1))

        palm_rx = float(rng.uniform(0.14, 0.19) * size)
        palm_ry = float(rng.uniform(0.18, 0.24) * size)
        thumb_rx = float(rng.uniform(0.07, 0.10) * size)
        thumb_ry = float(rng.uniform(0.10, 0.14) * size)
        thumb_dx = float(rng.uniform(0.12, 0.18) * size) * (-1.0 if label == 0 else 1.0)
        thumb_dy = float(rng.uniform(-0.05, 0.05) * size)

        return label, {
            "center_x": center_x,
            "center_y": center_y,
            "palm_rx": palm_rx,
            "palm_ry": palm_ry,
            "thumb_rx": thumb_rx,
            "thumb_ry": thumb_ry,
            "thumb_dx": thumb_dx,
            "thumb_dy": thumb_dy,
        }

    def __getitem__(self, idx: int):
        import torch

        size = int(self.cfg.image_size)
        label, params = self._sample_params(int(idx))
        rng = np.random.default_rng(int(self.cfg.seed) * 97_409 + int(idx))

        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        image = np.full((size, size), 0.06, dtype=np.float32)
        image += 0.03 * (1.0 - yy / max(float(size - 1), 1.0))

        cx = float(params["center_x"])
        cy = float(params["center_y"])
        palm = ((xx - cx) / float(params["palm_rx"])) ** 2 + ((yy - cy) / float(params["palm_ry"])) ** 2
        palm_mask = np.exp(-palm).astype(np.float32)

        tx = cx + float(params["thumb_dx"])
        ty = cy + float(params["thumb_dy"])
        thumb = ((xx - tx) / float(params["thumb_rx"])) ** 2 + ((yy - ty) / float(params["thumb_ry"])) ** 2
        thumb_mask = np.exp(-thumb).astype(np.float32)

        # Boost the thumb region so handedness is visually separable in a synthetic setting.
        image += 0.70 * palm_mask
        image += 0.55 * thumb_mask

        # Add a subtle wrist band to reduce degenerate solutions.
        wrist_y = cy + float(params["palm_ry"]) * 0.75
        image += 0.07 * np.exp(-((yy - wrist_y) ** 2) / (2.0 * 2.0 * 2.0)).astype(np.float32)

        image += rng.normal(0.0, float(self.cfg.noise_std), size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        return torch.from_numpy(image).unsqueeze(0), int(label)


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticHandednessDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticHandednessDataset", "get_dataloaders"]

