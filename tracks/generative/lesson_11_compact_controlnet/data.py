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


class SyntheticControlNetDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._targets, self._structure_hints = _make_synthetic_controlnet_data(cfg)

    def __len__(self) -> int:
        return int(self._targets.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._targets[i], self._structure_hints[i]


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
        image[0, fg] = 0.9
    elif mode == 1:
        y1 = int(torch.randint(2, image_size // 2, (1,), generator=generator).item())
        y2 = int(torch.randint(image_size // 2, image_size - 2, (1,), generator=generator).item())
        x1 = int(torch.randint(2, image_size // 2, (1,), generator=generator).item())
        x2 = int(torch.randint(image_size // 2, image_size - 2, (1,), generator=generator).item())
        image[:, y1:y2, x1:x2] = 0.85
    else:
        center = int(torch.randint(5, image_size - 5, (1,), generator=generator).item())
        thickness = int(torch.randint(1, 3, (1,), generator=generator).item())
        image[:, center - thickness : center + thickness + 1, :] = 0.9
        image[:, :, center - thickness : center + thickness + 1] = 0.9

    jitter = 0.04 * torch.rand((1, image_size, image_size), generator=generator, dtype=torch.float32)
    return torch.clamp(image + jitter, 0.0, 1.0)


def _structure_hint(target: torch.Tensor) -> torch.Tensor:
    gx = torch.abs(target[:, :, 1:] - target[:, :, :-1])
    gy = torch.abs(target[:, 1:, :] - target[:, :-1, :])
    edge_x = torch.nn.functional.pad(gx, (0, 1, 0, 0))
    edge_y = torch.nn.functional.pad(gy, (0, 0, 0, 1))
    edge = torch.clamp(edge_x + edge_y, 0.0, 1.0)
    return edge


def _make_synthetic_controlnet_data(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(int(cfg.seed))
    targets: list[torch.Tensor] = []
    structure_hints: list[torch.Tensor] = []

    for _ in range(int(cfg.num_samples)):
        target = _paint_shape(int(cfg.image_size), g)
        hint = _structure_hint(target)
        targets.append(target)
        structure_hints.append(hint)

    return torch.stack(targets, dim=0), torch.stack(structure_hints, dim=0)


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds: Dataset = SyntheticControlNetDataset(cfg)
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
