from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 8
    image_size: int = 48
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    in_channels: int = 3
    num_thing_classes: int = 3
    num_stuff_classes: int = 2
    max_instances: int = 2


class SyntheticPanopticSegmentationDataset:
    """Synthetic panoptic dataset with stuff regions + fixed-count rectangular thing instances."""

    def __init__(self, cfg: DataConfig) -> None:
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if int(cfg.in_channels) != 3:
            raise ValueError("This lesson expects in_channels == 3")
        if int(cfg.num_thing_classes) <= 0:
            raise ValueError("num_thing_classes must be > 0")
        if int(cfg.num_stuff_classes) <= 0:
            raise ValueError("num_stuff_classes must be > 0")
        if int(cfg.max_instances) <= 0:
            raise ValueError("max_instances must be > 0")
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int):
        import torch

        h = int(self.cfg.image_size)
        w = int(self.cfg.image_size)
        thing = int(self.cfg.num_thing_classes)
        stuff = int(self.cfg.num_stuff_classes)
        max_instances = int(self.cfg.max_instances)
        rng = np.random.default_rng(int(self.cfg.seed) * 1_000_003 + int(idx))

        semantic = np.zeros((h, w), dtype=np.int64)
        split_row = int(rng.integers(low=h // 3, high=(2 * h) // 3))
        semantic[split_row:, :] = 1 % stuff

        instance_masks = np.zeros((max_instances, h, w), dtype=np.float32)
        instance_classes = np.zeros((max_instances,), dtype=np.int64)

        for inst_id in range(max_instances):
            thing_class = int(rng.integers(low=0, high=thing))
            rect_h = int(rng.integers(low=max(4, h // 6), high=max(5, h // 3)))
            rect_w = int(rng.integers(low=max(4, w // 6), high=max(5, w // 3)))
            y0 = int(rng.integers(low=1, high=max(2, h - rect_h - 1)))
            x0 = int(rng.integers(low=1, high=max(2, w - rect_w - 1)))
            y1 = min(h, y0 + rect_h)
            x1 = min(w, x0 + rect_w)

            instance_masks[inst_id, y0:y1, x0:x1] = 1.0
            instance_classes[inst_id] = thing_class
            semantic[y0:y1, x0:x1] = stuff + thing_class

        palette = np.array(
            [
                [0.18, 0.20, 0.22],
                [0.26, 0.30, 0.36],
                [0.85, 0.28, 0.30],
                [0.24, 0.78, 0.38],
                [0.30, 0.48, 0.88],
                [0.82, 0.74, 0.26],
                [0.74, 0.32, 0.86],
                [0.20, 0.78, 0.84],
            ],
            dtype=np.float32,
        )
        image = palette[semantic % len(palette)].copy()
        image += 0.04 * rng.normal(size=(h, w, 3)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        return torch.from_numpy(image.transpose(2, 0, 1)).to(torch.float32), {
            "semantic_labels": torch.from_numpy(semantic).to(torch.long),
            "instance_masks": torch.from_numpy(instance_masks).to(torch.float32),
            "instance_classes": torch.from_numpy(instance_classes).to(torch.long),
        }


def get_dataloaders(cfg: DataConfig):
    from torch.utils.data import DataLoader, Subset

    dataset = SyntheticPanopticSegmentationDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticPanopticSegmentationDataset", "get_dataloaders"]

