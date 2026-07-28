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
    noise_std: float = 0.18
    impulse_prob: float = 0.015


class SyntheticDenoisingDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._noisy, self._clean = _make_synthetic_denoising_data(cfg)

    def __len__(self) -> int:
        return int(self._clean.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._noisy[i], self._clean[i]


def _paint_shape(image_size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    image = torch.zeros((1, image_size, image_size), dtype=torch.float32)
    mode = int(torch.randint(low=0, high=4, size=(1,), generator=generator).item())

    if mode == 0:
        cx = float(torch.empty((1,)).uniform_(-0.45, 0.45, generator=generator).item())
        cy = float(torch.empty((1,)).uniform_(-0.45, 0.45, generator=generator).item())
        radius = float(torch.empty((1,)).uniform_(0.2, 0.5, generator=generator).item())
        fg = (xx - cx).pow(2) + (yy - cy).pow(2) <= radius**2
        image[0, fg] = 0.92
    elif mode == 1:
        y1 = int(torch.randint(2, image_size // 2, (1,), generator=generator).item())
        y2 = int(torch.randint(image_size // 2, image_size - 2, (1,), generator=generator).item())
        x1 = int(torch.randint(2, image_size // 2, (1,), generator=generator).item())
        x2 = int(torch.randint(image_size // 2, image_size - 2, (1,), generator=generator).item())
        image[:, y1:y2, x1:x2] = 0.88
    elif mode == 2:
        center = int(torch.randint(5, image_size - 5, (1,), generator=generator).item())
        thickness = int(torch.randint(1, 3, (1,), generator=generator).item())
        image[:, center - thickness : center + thickness + 1, :] = 0.9
        image[:, :, center - thickness : center + thickness + 1] = 0.9
    else:
        phase = float(torch.empty((1,)).uniform_(0.0, 6.28318530718, generator=generator).item())
        waves = 0.5 + 0.3 * torch.sin(5.0 * xx + phase) * torch.cos(4.0 * yy + phase)
        image[0] = torch.clamp(waves, 0.0, 1.0)

    jitter = 0.05 * torch.rand((1, image_size, image_size), generator=generator, dtype=torch.float32)
    return torch.clamp(image + jitter, 0.0, 1.0)


def _add_noise(
    clean: torch.Tensor,
    *,
    generator: torch.Generator,
    noise_std: float,
    impulse_prob: float,
) -> torch.Tensor:
    noisy = clean + float(noise_std) * torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    impulse_mask = torch.rand(clean.shape, generator=generator, dtype=clean.dtype) < float(impulse_prob)
    impulse_vals = torch.rand(clean.shape, generator=generator, dtype=clean.dtype)
    noisy = torch.where(impulse_mask, impulse_vals, noisy)
    return torch.clamp(noisy, 0.0, 1.0)


def _make_synthetic_denoising_data(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(int(cfg.seed))
    noisy: list[torch.Tensor] = []
    clean: list[torch.Tensor] = []

    for _ in range(int(cfg.num_samples)):
        clean_img = _paint_shape(int(cfg.image_size), g)
        noisy_img = _add_noise(
            clean_img,
            generator=g,
            noise_std=float(cfg.noise_std),
            impulse_prob=float(cfg.impulse_prob),
        )
        clean.append(clean_img)
        noisy.append(noisy_img)

    return torch.stack(noisy, dim=0), torch.stack(clean, dim=0)


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds: Dataset = SyntheticDenoisingDataset(cfg)
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
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    return train_loader, val_loader
