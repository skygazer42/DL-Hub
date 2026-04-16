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
    num_classes: int = 6
    noise_std: float = 0.02


class SyntheticFingerCountDataset:
    """Render a compact synthetic hand crop and classify the number of raised fingers."""

    class_names = ("zero", "one", "two", "three", "four", "five")

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 48:
            raise ValueError("image_size must be >= 48")
        if int(cfg.in_channels) != 1:
            raise ValueError("This lesson expects grayscale inputs.")
        if int(cfg.num_classes) != 6:
            raise ValueError("This lesson uses exactly 6 classes.")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _sample_params(self, idx: int) -> tuple[int, dict[str, float]]:
        rng = np.random.default_rng(int(self.cfg.seed) * 1_000_003 + int(idx))
        label = int((int(idx) + int(self.cfg.seed)) % int(self.cfg.num_classes))
        size = int(self.cfg.image_size)

        center_x = float(rng.uniform(0.46, 0.54) * (size - 1))
        center_y = float(rng.uniform(0.58, 0.66) * (size - 1))
        palm_rx = float(rng.uniform(0.14, 0.18) * size)
        palm_ry = float(rng.uniform(0.16, 0.22) * size)
        finger_width = float(rng.uniform(0.035, 0.05) * size)
        finger_height = float(rng.uniform(0.16, 0.23) * size)
        finger_spacing = float(rng.uniform(0.07, 0.09) * size)

        return label, {
            "center_x": center_x,
            "center_y": center_y,
            "palm_rx": palm_rx,
            "palm_ry": palm_ry,
            "finger_width": finger_width,
            "finger_height": finger_height,
            "finger_spacing": finger_spacing,
        }

    def __getitem__(self, idx: int):
        import torch

        label, params = self._sample_params(int(idx))
        size = int(self.cfg.image_size)
        rng = np.random.default_rng(int(self.cfg.seed) * 97_409 + int(idx))

        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        image = np.full((size, size), 0.05, dtype=np.float32)
        image += 0.03 * (1.0 - yy / max(float(size - 1), 1.0))

        cx = float(params["center_x"])
        cy = float(params["center_y"])
        palm = ((xx - cx) / float(params["palm_rx"])) ** 2 + ((yy - cy) / float(params["palm_ry"])) ** 2
        palm_mask = np.exp(-palm).astype(np.float32)
        image += 0.70 * palm_mask

        wrist_y = cy + float(params["palm_ry"]) * 0.9
        image += 0.08 * np.exp(-((yy - wrist_y) ** 2) / (2.0 * 2.4 * 2.4)).astype(np.float32)

        count = int(label)
        if count == 0:
            knuckle_y = cy - float(params["palm_ry"]) * 0.75
            folded = np.exp(-((yy - knuckle_y) ** 2) / (2.0 * 1.8 * 1.8)).astype(np.float32)
            folded *= np.exp(-((xx - cx) ** 2) / (2.0 * (float(params["palm_rx"]) * 0.8) ** 2)).astype(np.float32)
            image += 0.18 * folded
        else:
            offset_center = 0.5 * float(count - 1)
            finger_base_y = cy - float(params["palm_ry"]) * 1.05
            for finger_idx in range(count):
                x_offset = (float(finger_idx) - offset_center) * float(params["finger_spacing"])
                fx = cx + x_offset + float(rng.uniform(-0.02, 0.02) * size)
                fy = finger_base_y - float(rng.uniform(0.15, 0.35) * float(params["finger_height"]))
                width = float(params["finger_width"]) * float(rng.uniform(0.9, 1.1))
                height = float(params["finger_height"]) * float(rng.uniform(0.9, 1.1))

                finger = np.exp(-((xx - fx) ** 2) / (2.0 * width * width)).astype(np.float32)
                finger *= np.exp(-((yy - fy) ** 2) / (2.0 * height * height)).astype(np.float32)
                tip = np.exp(
                    -((xx - fx) ** 2 + (yy - (fy - 0.35 * height)) ** 2) / (2.0 * (width * 0.9) ** 2)
                ).astype(np.float32)
                image += 0.58 * finger
                image += 0.16 * tip

        image += rng.normal(0.0, float(self.cfg.noise_std), size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        return torch.from_numpy(image).unsqueeze(0), int(label)


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticFingerCountDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticFingerCountDataset", "get_dataloaders"]
