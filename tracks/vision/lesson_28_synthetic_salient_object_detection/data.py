from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 16
    image_size: int = 32
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0

    in_channels: int = 1
    min_object_size: int = 5
    max_object_size: int = 12
    min_distractors: int = 2
    max_distractors: int = 5
    noise_std: float = 0.08


def _validate_config(cfg: DataConfig) -> None:
    if int(cfg.num_samples) <= 0:
        raise ValueError("num_samples must be > 0")
    if int(cfg.batch_size) <= 0:
        raise ValueError("batch_size must be > 0")
    if int(cfg.image_size) < 16:
        raise ValueError("image_size must be >= 16")
    if not (0.0 < float(cfg.val_fraction) < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")
    if int(cfg.in_channels) != 1:
        raise ValueError("in_channels must be 1 for v1")
    if int(cfg.min_object_size) < 3 or int(cfg.max_object_size) < int(cfg.min_object_size):
        raise ValueError("invalid object size range")
    if int(cfg.min_distractors) < 0 or int(cfg.max_distractors) < int(cfg.min_distractors):
        raise ValueError("invalid distractor count range")


def _draw_rectangle(canvas: torch.Tensor, top: int, left: int, height: int, width: int, value: float) -> None:
    canvas[top : top + height, left : left + width] = float(value)


def _draw_circle(canvas: torch.Tensor, cy: int, cx: int, radius: int, value: float) -> None:
    size = int(canvas.shape[-1])
    yy = torch.arange(size, dtype=torch.float32).view(size, 1)
    xx = torch.arange(size, dtype=torch.float32).view(1, size)
    mask = ((yy - float(cy)) ** 2 + (xx - float(cx)) ** 2) <= float(radius * radius)
    canvas[mask] = float(value)


def _draw_shape(
    canvas: torch.Tensor,
    *,
    generator: torch.Generator,
    size_range: tuple[int, int],
    value_range: tuple[float, float],
) -> torch.Tensor:
    size = int(canvas.shape[-1])
    low, high = int(size_range[0]), int(size_range[1])
    obj_h = int(torch.randint(low, high + 1, (1,), generator=generator).item())
    obj_w = int(torch.randint(low, high + 1, (1,), generator=generator).item())
    top = int(torch.randint(0, size - obj_h + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, size - obj_w + 1, (1,), generator=generator).item())
    value = float(torch.empty((), dtype=torch.float32).uniform_(value_range[0], value_range[1], generator=generator).item())

    mask = torch.zeros_like(canvas)
    if bool(torch.randint(0, 2, (1,), generator=generator).item()):
        _draw_rectangle(canvas, top, left, obj_h, obj_w, value)
        _draw_rectangle(mask, top, left, obj_h, obj_w, 1.0)
    else:
        radius = max(2, min(obj_h, obj_w) // 2)
        cy = min(size - radius - 1, top + obj_h // 2)
        cx = min(size - radius - 1, left + obj_w // 2)
        _draw_circle(canvas, cy, cx, radius, value)
        _draw_circle(mask, cy, cx, radius, 1.0)
    return mask


def make_salient_sample(cfg: DataConfig, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_config(cfg)
    size = int(cfg.image_size)
    generator = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + int(idx))

    image = torch.randn((size, size), generator=generator, dtype=torch.float32) * float(cfg.noise_std)
    image += 0.12

    distractors = int(
        torch.randint(
            int(cfg.min_distractors),
            int(cfg.max_distractors) + 1,
            (1,),
            generator=generator,
        ).item()
    )
    for _ in range(distractors):
        _draw_shape(
            image,
            generator=generator,
            size_range=(int(cfg.min_object_size), int(cfg.max_object_size)),
            value_range=(0.25, 0.65),
        )

    salient_mask = _draw_shape(
        image,
        generator=generator,
        size_range=(int(cfg.min_object_size) + 1, int(cfg.max_object_size) + 2),
        value_range=(0.8, 1.0),
    )
    image = image.clamp(0.0, 1.0).unsqueeze(0)
    return image, salient_mask.unsqueeze(0)


class SyntheticSalientObjectDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return make_salient_sample(self.cfg, int(idx))


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    _validate_config(cfg)
    ds = SyntheticSalientObjectDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(ds),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )
    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=False,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=False,
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "SyntheticSalientObjectDataset", "get_dataloaders", "make_salient_sample"]

