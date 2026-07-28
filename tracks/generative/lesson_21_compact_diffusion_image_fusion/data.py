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


class SyntheticImageFusionDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._obs_a, self._obs_b, self._target = _make_synthetic_image_fusion_data(cfg)

    def __len__(self) -> int:
        return int(self._target.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._obs_a[i], self._obs_b[i], self._target[i]


def _paint_shape(image_size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    image = torch.zeros((1, image_size, image_size), dtype=torch.float32)
    mode = int(torch.randint(low=0, high=4, size=(1,), generator=generator).item())

    if mode == 0:
        cx = float(torch.empty((1,)).uniform_(-0.5, 0.5, generator=generator).item())
        cy = float(torch.empty((1,)).uniform_(-0.5, 0.5, generator=generator).item())
        radius = float(torch.empty((1,)).uniform_(0.2, 0.55, generator=generator).item())
        fg = (xx - cx).pow(2) + (yy - cy).pow(2) <= radius**2
        image[0, fg] = 0.9
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
        slope = float(torch.empty((1,)).uniform_(-0.8, 0.8, generator=generator).item())
        bias = float(torch.empty((1,)).uniform_(-0.3, 0.3, generator=generator).item())
        width = float(torch.empty((1,)).uniform_(0.1, 0.28, generator=generator).item())
        diag = torch.abs(yy - slope * xx - bias) <= width
        image[0, diag] = 0.86

    jitter = 0.05 * torch.rand((1, image_size, image_size), generator=generator, dtype=torch.float32)
    return torch.clamp(image + jitter, 0.0, 1.0)


def _gaussian_blur(image: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        dtype=image.dtype,
        device=image.device,
    )
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3)
    return torch.nn.functional.conv2d(image.unsqueeze(0), kernel, padding=1).squeeze(0)


def _complementary_observations(target: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    size = int(target.shape[-1])
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, size, dtype=target.dtype),
        torch.linspace(-1.0, 1.0, size, dtype=target.dtype),
        indexing="ij",
    )
    freq_a = float(torch.empty((1,)).uniform_(5.0, 9.0, generator=generator).item())
    freq_b = float(torch.empty((1,)).uniform_(5.0, 9.0, generator=generator).item())
    phase_a = float(torch.empty((1,)).uniform_(-3.14, 3.14, generator=generator).item())
    phase_b = float(torch.empty((1,)).uniform_(-3.14, 3.14, generator=generator).item())

    mask_a = (torch.sin(freq_a * xx + phase_a) > 0).to(dtype=target.dtype) * 0.75 + 0.15
    mask_b = (torch.cos(freq_b * yy + phase_b) > 0).to(dtype=target.dtype) * 0.75 + 0.15
    blurred = _gaussian_blur(target)
    noise_a = 0.025 * torch.randn(target.shape, generator=generator, dtype=target.dtype)
    noise_b = 0.025 * torch.randn(target.shape, generator=generator, dtype=target.dtype)

    obs_a = target * mask_a.unsqueeze(0) + 0.2 * blurred + 0.05 + noise_a
    obs_b = target * mask_b.unsqueeze(0) + 0.2 * torch.roll(blurred, shifts=1, dims=2) + 0.05 + noise_b
    return obs_a.clamp(0.0, 1.0), obs_b.clamp(0.0, 1.0)


def _make_synthetic_image_fusion_data(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(int(cfg.seed))
    obs_a: list[torch.Tensor] = []
    obs_b: list[torch.Tensor] = []
    target: list[torch.Tensor] = []

    for _ in range(int(cfg.num_samples)):
        clean = _paint_shape(int(cfg.image_size), g)
        view_a, view_b = _complementary_observations(clean, g)
        obs_a.append(view_a)
        obs_b.append(view_b)
        target.append(clean)

    return torch.stack(obs_a, dim=0), torch.stack(obs_b, dim=0), torch.stack(target, dim=0)


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds: Dataset = SyntheticImageFusionDataset(cfg)
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
