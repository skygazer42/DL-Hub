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


class SyntheticDerainingDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._rainy, self._clean = _make_synthetic_deraining_data(cfg)

    def __len__(self) -> int:
        return int(self._clean.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._rainy[i], self._clean[i]


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
        radius = float(torch.empty((1,)).uniform_(0.25, 0.55, generator=generator).item())
        fg = (xx - cx).pow(2) + (yy - cy).pow(2) <= radius**2
        image[0, fg] = 0.92
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
    return torch.clamp(image + jitter, 0.0, 1.0)


def _add_rain_streaks(clean: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    _, h, w = clean.shape
    rain = torch.zeros_like(clean)
    streak_count = int(torch.randint(10, 22, (1,), generator=generator).item())
    length_min = max(4, h // 6)
    length_max = max(length_min + 1, h // 2)

    for _ in range(streak_count):
        x0 = int(torch.randint(0, w, (1,), generator=generator).item())
        y0 = int(torch.randint(0, h, (1,), generator=generator).item())
        length = int(torch.randint(length_min, length_max + 1, (1,), generator=generator).item())
        thickness = int(torch.randint(1, 3, (1,), generator=generator).item())
        slope = int(torch.randint(-1, 2, (1,), generator=generator).item())
        strength = float(torch.empty((1,)).uniform_(0.25, 0.55, generator=generator).item())

        for step in range(length):
            y = y0 + step
            if y >= h:
                break
            x = x0 + slope * step
            if x < 0 or x >= w:
                continue
            x1 = max(0, x - thickness)
            x2 = min(w, x + thickness + 1)
            rain[:, y : y + 1, x1:x2] = torch.maximum(rain[:, y : y + 1, x1:x2], torch.tensor(strength))

    blur_kernel = torch.tensor(
        [[0.05, 0.10, 0.05], [0.10, 0.40, 0.10], [0.05, 0.10, 0.05]],
        dtype=clean.dtype,
        device=clean.device,
    ).view(1, 1, 3, 3)
    rain_soft = torch.nn.functional.conv2d(rain.unsqueeze(0), blur_kernel, padding=1).squeeze(0)
    haze_veil = torch.empty((1, 1, 1), dtype=clean.dtype).uniform_(0.02, 0.08, generator=generator)
    sensor_noise = 0.02 * torch.randn(clean.shape, generator=generator, dtype=clean.dtype)
    rainy = clean * 0.9 + rain_soft + haze_veil + sensor_noise
    return rainy.clamp(0.0, 1.0)


def _make_synthetic_deraining_data(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(int(cfg.seed))
    rainy: list[torch.Tensor] = []
    clean: list[torch.Tensor] = []

    for _ in range(int(cfg.num_samples)):
        clean_img = _paint_shape(int(cfg.image_size), g)
        rainy_img = _add_rain_streaks(clean_img, g)
        clean.append(clean_img)
        rainy.append(rainy_img)

    return torch.stack(rainy, dim=0), torch.stack(clean, dim=0)


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds: Dataset = SyntheticDerainingDataset(cfg)
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

