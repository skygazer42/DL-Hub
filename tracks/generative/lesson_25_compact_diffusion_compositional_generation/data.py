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
    mode = int(torch.randint(0, 4, (1,), generator=generator).item())

    if mode == 0:
        cx = float(torch.empty((1,)).uniform_(-0.4, 0.4, generator=generator).item())
        cy = float(torch.empty((1,)).uniform_(-0.4, 0.4, generator=generator).item())
        radius = float(torch.empty((1,)).uniform_(0.22, 0.48, generator=generator).item())
        fg = (xx - cx).pow(2) + (yy - cy).pow(2) <= radius**2
        mask[0, fg] = 1.0
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
        slope = float(torch.empty((1,)).uniform_(-0.9, 0.9, generator=generator).item())
        bias = float(torch.empty((1,)).uniform_(-0.35, 0.35, generator=generator).item())
        width = float(torch.empty((1,)).uniform_(0.1, 0.24, generator=generator).item())
        diag = torch.abs(yy - slope * xx - bias) <= width
        mask[0, diag] = 1.0
    return mask


def _style_texture(image_size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    mode = int(torch.randint(0, 3, (1,), generator=generator).item())
    if mode == 0:
        phase = float(torch.empty((1,)).uniform_(-3.14, 3.14, generator=generator).item())
        tex = 0.5 + 0.5 * torch.sin(xx * 9.0 + phase)
    elif mode == 1:
        tex = (((xx * 7.0).floor() + (yy * 7.0).floor()) % 2.0 == 0.0).to(torch.float32)
    else:
        tex = 0.5 + 0.5 * torch.cos((xx.pow(2) + yy.pow(2)).sqrt() * 11.0)
    tex = tex.unsqueeze(0)
    tex += 0.04 * torch.rand((1, image_size, image_size), generator=generator, dtype=torch.float32)
    return tex.clamp(0.0, 1.0)


def _make_structure(mask: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    noise = 0.06 * torch.rand(mask.shape, generator=generator, dtype=torch.float32)
    return (0.1 + 0.78 * mask + noise).clamp(0.0, 1.0)


def _make_style(texture: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    style = 0.12 + 0.8 * texture
    style += 0.03 * torch.rand(texture.shape, generator=generator, dtype=torch.float32)
    return style.clamp(0.0, 1.0)


def _compose_target(mask: torch.Tensor, style: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    background = 0.06 + 0.05 * torch.rand(mask.shape, generator=generator, dtype=torch.float32)
    target = background * (1.0 - mask) + style * mask
    target += 0.03 * torch.rand(mask.shape, generator=generator, dtype=torch.float32)
    return target.clamp(0.0, 1.0)


class SyntheticCompositionalGenerationDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._structure, self._style, self._target = _make_synthetic_compositional_data(cfg)

    def __len__(self) -> int:
        return int(self._target.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._structure[i], self._style[i], self._target[i]


def _make_synthetic_compositional_data(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(cfg.seed))
    structures: list[torch.Tensor] = []
    styles: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for _ in range(int(cfg.num_samples)):
        mask = _shape_mask(int(cfg.image_size), generator)
        texture = _style_texture(int(cfg.image_size), generator)
        structure = _make_structure(mask, generator)
        style = _make_style(texture, generator)
        target = _compose_target(mask, style, generator)
        structures.append(structure)
        styles.append(style)
        targets.append(target)

    return torch.stack(structures, dim=0), torch.stack(styles, dim=0), torch.stack(targets, dim=0)


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset: Dataset = SyntheticCompositionalGenerationDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticCompositionalGenerationDataset", "get_dataloaders"]
