from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 512
    batch_size: int = 8
    image_size: int = 32
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    in_channels: int = 3
    set_size: int = 3
    min_object_size: int = 6
    max_object_size: int = 14
    min_distractors: int = 1
    max_distractors: int = 3
    noise_std: float = 0.05


def _validate_config(cfg: DataConfig) -> None:
    if int(cfg.num_samples) <= 0:
        raise ValueError("num_samples must be > 0")
    if int(cfg.batch_size) <= 0:
        raise ValueError("batch_size must be > 0")
    if int(cfg.image_size) < 24:
        raise ValueError("image_size must be >= 24")
    if not (0.0 < float(cfg.val_fraction) < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")
    if int(cfg.in_channels) != 3:
        raise ValueError("in_channels must be 3 for this lesson")
    if int(cfg.set_size) < 2:
        raise ValueError("set_size must be >= 2")
    if int(cfg.min_object_size) < 4 or int(cfg.max_object_size) < int(cfg.min_object_size):
        raise ValueError("invalid object-size range")
    if int(cfg.min_distractors) < 0 or int(cfg.max_distractors) < int(cfg.min_distractors):
        raise ValueError("invalid distractor-count range")


def _draw_rect(mask: torch.Tensor, top: int, left: int, h: int, w: int, value: float) -> None:
    mask[top : top + h, left : left + w] = float(value)


def _draw_disk(mask: torch.Tensor, cy: int, cx: int, radius: int, value: float) -> None:
    size = int(mask.shape[-1])
    yy = torch.arange(size, dtype=torch.float32).view(size, 1)
    xx = torch.arange(size, dtype=torch.float32).view(1, size)
    inside = ((yy - float(cy)) ** 2 + (xx - float(cx)) ** 2) <= float(radius * radius)
    mask[inside] = float(value)


def _shared_object_mask(cfg: DataConfig, *, generator: torch.Generator) -> torch.Tensor:
    size = int(cfg.image_size)
    low = int(cfg.min_object_size)
    high = int(cfg.max_object_size)
    obj_h = int(torch.randint(low, high + 1, (1,), generator=generator).item())
    obj_w = int(torch.randint(low, high + 1, (1,), generator=generator).item())
    top = int(torch.randint(2, size - obj_h - 1, (1,), generator=generator).item())
    left = int(torch.randint(2, size - obj_w - 1, (1,), generator=generator).item())
    mask = torch.zeros((size, size), dtype=torch.float32)
    if bool(torch.randint(0, 2, (1,), generator=generator).item()):
        _draw_rect(mask, top, left, obj_h, obj_w, 1.0)
    else:
        radius = max(2, min(obj_h, obj_w) // 2)
        cy = min(size - radius - 1, top + obj_h // 2)
        cx = min(size - radius - 1, left + obj_w // 2)
        _draw_disk(mask, cy, cx, radius, 1.0)
    return mask


def _paint_mask(image: torch.Tensor, mask: torch.Tensor, color: torch.Tensor) -> None:
    for channel in range(int(image.shape[0])):
        image[channel] = torch.where(mask > 0.5, color[channel], image[channel])


def _jitter_mask(mask: torch.Tensor, *, shift_y: int, shift_x: int) -> torch.Tensor:
    return torch.roll(mask, shifts=(int(shift_y), int(shift_x)), dims=(0, 1))


def make_co_segmentation_sample(cfg: DataConfig, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _validate_config(cfg)
    generator = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(idx))
    size = int(cfg.image_size)
    set_size = int(cfg.set_size)
    images = []
    masks = []

    shared_mask = _shared_object_mask(cfg, generator=generator)
    shared_color = torch.tensor([0.85, 0.30, 0.25], dtype=torch.float32)

    for _ in range(set_size):
        image = torch.randn((3, size, size), generator=generator, dtype=torch.float32) * float(cfg.noise_std) + 0.10
        shift_y = int(torch.randint(-2, 3, (1,), generator=generator).item())
        shift_x = int(torch.randint(-2, 3, (1,), generator=generator).item())
        mask = _jitter_mask(shared_mask, shift_y=shift_y, shift_x=shift_x)
        _paint_mask(image, mask, shared_color)

        num_distractors = int(
            torch.randint(
                int(cfg.min_distractors),
                int(cfg.max_distractors) + 1,
                (1,),
                generator=generator,
            ).item()
        )
        for _ in range(num_distractors):
            shape_mask = _shared_object_mask(cfg, generator=generator)
            distractor_color = torch.empty((3,), dtype=torch.float32).uniform_(0.30, 0.80, generator=generator)
            _paint_mask(image, shape_mask * (1.0 - mask), distractor_color)

        image = image.clamp(0.0, 1.0)
        images.append(image)
        masks.append(mask)

    mask_tensor = torch.stack(masks, dim=0)
    class_index = (mask_tensor > 0.5).to(torch.long)
    return torch.stack(images, dim=0), {"mask": mask_tensor, "class_index": class_index}


class SyntheticCoSegmentationDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return make_co_segmentation_sample(self.cfg, int(idx))


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    _validate_config(cfg)
    dataset = SyntheticCoSegmentationDataset(cfg)
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
        pin_memory=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=False,
    )
    return train_loader, val_loader


__all__ = [
    "DataConfig",
    "SyntheticCoSegmentationDataset",
    "get_dataloaders",
    "make_co_segmentation_sample",
]
