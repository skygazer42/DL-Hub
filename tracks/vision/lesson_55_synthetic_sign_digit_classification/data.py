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
    num_classes: int = 10
    noise_std: float = 0.02


class SyntheticSignDigitDataset:
    """Render a compact synthetic hand crop and classify a synthetic sign-digit (0..9).

    This is a teaching-friendly proxy for hand-sign digit classification: a palm blob plus a
    small number of finger-like blobs, combined with a digit-specific corner marker.
    """

    class_names = tuple(str(i) for i in range(10))

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 48:
            raise ValueError("image_size must be >= 48")
        if int(cfg.in_channels) != 1:
            raise ValueError("This lesson expects grayscale inputs.")
        if int(cfg.num_classes) != 10:
            raise ValueError("This lesson uses exactly 10 classes (digits 0..9).")
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
        finger_w = float(rng.uniform(0.035, 0.05) * size)
        finger_h = float(rng.uniform(0.14, 0.22) * size)
        finger_spacing = float(rng.uniform(0.07, 0.09) * size)

        return label, {
            "center_x": center_x,
            "center_y": center_y,
            "palm_rx": palm_rx,
            "palm_ry": palm_ry,
            "finger_w": finger_w,
            "finger_h": finger_h,
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
        image += 0.68 * palm_mask

        # Digit -> "raised finger count" proxy: 0..4 (label % 5) plus a style group (label // 5).
        finger_count = int(label % 5)
        style_group = int(label // 5)  # 0 or 1
        finger_base_y = cy - float(params["palm_ry"]) * 1.05
        if finger_count > 0:
            offset_center = 0.5 * float(finger_count - 1)
            for k in range(finger_count):
                x_offset = (float(k) - offset_center) * float(params["finger_spacing"])
                fx = cx + x_offset + float(rng.uniform(-0.02, 0.02) * size)
                fy = finger_base_y - float(rng.uniform(0.10, 0.35) * float(params["finger_h"]))
                w = float(params["finger_w"]) * float(rng.uniform(0.9, 1.1))
                h = float(params["finger_h"]) * float(rng.uniform(0.9, 1.1))
                finger = np.exp(-((xx - fx) ** 2) / (2.0 * w * w)).astype(np.float32)
                finger *= np.exp(-((yy - fy) ** 2) / (2.0 * h * h)).astype(np.float32)
                image += 0.55 * finger

        # Add a digit-specific "marker" (two corners, five slots each) to make 10-way classification stable.
        slot = int(label % 5)
        if style_group == 0:
            mx = float(rng.uniform(0.08, 0.14) * size)
        else:
            mx = float(rng.uniform(0.86, 0.92) * size)
        my = float((0.12 + 0.14 * slot) * size)
        marker = np.exp(-((xx - mx) ** 2 + (yy - my) ** 2) / (2.0 * 1.4 * 1.4)).astype(np.float32)
        image += 0.22 * marker

        # Wrist band for extra structure.
        wrist_y = cy + float(params["palm_ry"]) * 0.9
        image += 0.08 * np.exp(-((yy - wrist_y) ** 2) / (2.0 * 2.4 * 2.4)).astype(np.float32)

        image += rng.normal(0.0, float(self.cfg.noise_std), size=(size, size)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)
        return torch.from_numpy(image).unsqueeze(0), int(label)


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticSignDigitDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticSignDigitDataset", "get_dataloaders"]
