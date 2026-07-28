from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 32
    image_size: int = 28
    upscale_factor: int = 2
    num_workers: int = 0
    num_samples: int = 256
    seed: int = 0
    val_fraction: float = 0.2


class SyntheticSuperResolutionDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._low_res, self._high_res = _make_synthetic_super_resolution_data(cfg)

    def __len__(self) -> int:
        return int(self._high_res.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._low_res[i], self._high_res[i]


def _paint_shape(image_size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    image = torch.zeros((1, image_size, image_size), dtype=torch.float32)
    mode = int(torch.randint(low=0, high=3, size=(1,), generator=generator).item())

    if mode == 0:
        cx = float(torch.empty((1,)).uniform_(-0.5, 0.5, generator=generator).item())
        cy = float(torch.empty((1,)).uniform_(-0.5, 0.5, generator=generator).item())
        radius = float(torch.empty((1,)).uniform_(0.22, 0.52, generator=generator).item())
        fg = (xx - cx).pow(2) + (yy - cy).pow(2) <= radius**2
        image[0, fg] = 0.9
    elif mode == 1:
        y1 = int(torch.randint(2, image_size // 2, (1,), generator=generator).item())
        y2 = int(torch.randint(image_size // 2, image_size - 2, (1,), generator=generator).item())
        x1 = int(torch.randint(2, image_size // 2, (1,), generator=generator).item())
        x2 = int(torch.randint(image_size // 2, image_size - 2, (1,), generator=generator).item())
        image[:, y1:y2, x1:x2] = 0.88
    else:
        center = int(torch.randint(5, image_size - 5, (1,), generator=generator).item())
        thickness = int(torch.randint(1, 3, (1,), generator=generator).item())
        image[:, center - thickness : center + thickness + 1, :] = 0.9
        image[:, :, center - thickness : center + thickness + 1] = 0.9

    jitter = 0.04 * torch.rand((1, image_size, image_size), generator=generator, dtype=torch.float32)
    return (image + jitter).clamp_(0.0, 1.0)


def _degrade_to_low_res(high_res: torch.Tensor, upscale_factor: int) -> torch.Tensor:
    if upscale_factor <= 0:
        raise ValueError(f"upscale_factor must be positive, got {upscale_factor}")
    if high_res.shape[-1] % upscale_factor != 0:
        raise ValueError("high_res width must be divisible by upscale_factor")
    if high_res.shape[-2] % upscale_factor != 0:
        raise ValueError("high_res height must be divisible by upscale_factor")

    low_h = high_res.shape[-2] // upscale_factor
    low_w = high_res.shape[-1] // upscale_factor
    x = high_res.unsqueeze(0)
    x = F.interpolate(x, size=(low_h, low_w), mode="area")
    return x.squeeze(0).clamp_(0.0, 1.0)


def _make_synthetic_super_resolution_data(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor]:
    size = int(cfg.image_size)
    upscale = int(cfg.upscale_factor)
    if size % upscale != 0:
        raise ValueError(f"image_size ({size}) must be divisible by upscale_factor ({upscale})")

    g = torch.Generator().manual_seed(int(cfg.seed))
    low_res: list[torch.Tensor] = []
    high_res: list[torch.Tensor] = []
    for _ in range(int(cfg.num_samples)):
        hr = _paint_shape(size, g)
        lr = _degrade_to_low_res(hr, upscale)
        high_res.append(hr)
        low_res.append(lr)

    return torch.stack(low_res, dim=0), torch.stack(high_res, dim=0)


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds: Dataset = SyntheticSuperResolutionDataset(cfg)
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
