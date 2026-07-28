from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 32
    image_size: int = 28
    num_workers: int = 0
    num_samples: int = 256
    seed: int = 0
    val_fraction: float = 0.2


def _shape_mask(image_size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    mask = torch.zeros((1, image_size, image_size), dtype=torch.float32)
    mode = int(torch.randint(low=0, high=4, size=(1,), generator=generator).item())

    if mode == 0:
        cx = float(torch.empty((1,)).uniform_(-0.45, 0.45, generator=generator).item())
        cy = float(torch.empty((1,)).uniform_(-0.45, 0.45, generator=generator).item())
        radius = float(torch.empty((1,)).uniform_(0.2, 0.5, generator=generator).item())
        mask[0, (xx - cx).pow(2) + (yy - cy).pow(2) <= radius**2] = 1.0
    elif mode == 1:
        y1 = int(torch.randint(2, image_size // 2, (1,), generator=generator).item())
        y2 = int(torch.randint(image_size // 2, image_size - 2, (1,), generator=generator).item())
        x1 = int(torch.randint(2, image_size // 2, (1,), generator=generator).item())
        x2 = int(torch.randint(image_size // 2, image_size - 2, (1,), generator=generator).item())
        mask[:, y1:y2, x1:x2] = 1.0
    elif mode == 2:
        center = int(torch.randint(5, image_size - 5, (1,), generator=generator).item())
        thickness = int(torch.randint(1, 3, (1,), generator=generator).item())
        mask[:, center - thickness : center + thickness + 1, :] = 1.0
        mask[:, :, center - thickness : center + thickness + 1] = 1.0
    else:
        slope = float(torch.empty((1,)).uniform_(-0.85, 0.85, generator=generator).item())
        bias = float(torch.empty((1,)).uniform_(-0.3, 0.3, generator=generator).item())
        width = float(torch.empty((1,)).uniform_(0.09, 0.24, generator=generator).item())
        mask[0, torch.abs(yy - slope * xx - bias) <= width] = 1.0

    return mask


def _texture(image_size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    mode = int(torch.randint(0, 3, (1,), generator=generator).item())
    if mode == 0:
        tex = 0.5 + 0.5 * torch.sin(xx * 8.0 + yy * 3.0)
    elif mode == 1:
        tex = (((xx * 8.0).floor() + (yy * 8.0).floor()) % 2.0 == 0.0).to(torch.float32)
    else:
        tex = 0.5 + 0.5 * torch.cos((xx.pow(2) + yy.pow(2)).sqrt() * 10.0)
    tex = tex.unsqueeze(0)
    tex += 0.03 * torch.rand((1, image_size, image_size), generator=generator, dtype=torch.float32)
    return tex.clamp(0.0, 1.0)


def _make_target(mask: torch.Tensor, texture: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    bg = 0.05 + 0.08 * torch.rand(mask.shape, generator=generator, dtype=torch.float32)
    fg = 0.2 + 0.72 * texture
    target = bg * (1.0 - mask) + fg * mask
    target += 0.02 * torch.rand(mask.shape, generator=generator, dtype=torch.float32)
    return target.clamp(0.0, 1.0)


def _smooth_image(image: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        dtype=image.dtype,
    )
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3)
    return torch.nn.functional.conv2d(image.unsqueeze(0), kernel, padding=1).squeeze(0)


def _make_source(target: torch.Tensor, mask: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    shifted = torch.roll(target, shifts=1, dims=2)
    blurred = _smooth_image(_smooth_image(target))
    mask_soft = (0.35 + 0.55 * mask).clamp(0.0, 1.0)
    source = mask_soft * shifted + (1.0 - mask_soft) * blurred
    source += 0.04 * torch.randn(target.shape, generator=generator, dtype=torch.float32)
    source += 0.03 * torch.rand(target.shape, generator=generator, dtype=torch.float32)
    return source.clamp(0.0, 1.0)


def _make_synthetic_variation_data(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(cfg.seed))
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for _ in range(int(cfg.num_samples)):
        mask = _shape_mask(int(cfg.image_size), generator)
        texture = _texture(int(cfg.image_size), generator)
        target = _make_target(mask, texture, generator)
        source = _make_source(target, mask, generator)
        sources.append(source)
        targets.append(target)

    return torch.stack(sources, dim=0), torch.stack(targets, dim=0)


class SyntheticImageVariationDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._source, self._target = _make_synthetic_variation_data(cfg)

    def __len__(self) -> int:
        return int(self._target.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._source[i], self._target[i]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset: Dataset = SyntheticImageVariationDataset(cfg)
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
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "SyntheticImageVariationDataset", "get_dataloaders"]
