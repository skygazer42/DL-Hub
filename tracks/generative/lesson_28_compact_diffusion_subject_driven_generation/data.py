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
        slope = float(torch.empty((1,)).uniform_(-0.9, 0.9, generator=generator).item())
        bias = float(torch.empty((1,)).uniform_(-0.35, 0.35, generator=generator).item())
        width = float(torch.empty((1,)).uniform_(0.1, 0.24, generator=generator).item())
        mask[0, torch.abs(yy - slope * xx - bias) <= width] = 1.0
    return mask


def _subject_texture(image_size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    mode = int(torch.randint(0, 3, (1,), generator=generator).item())
    if mode == 0:
        phase = float(torch.empty((1,)).uniform_(-3.14, 3.14, generator=generator).item())
        texture = 0.5 + 0.5 * torch.sin(xx * 8.0 + yy * 3.0 + phase)
    elif mode == 1:
        texture = (((xx * 8.0).floor() + (yy * 8.0).floor()) % 2.0 == 0.0).to(torch.float32)
    else:
        radial = (xx.pow(2) + yy.pow(2)).sqrt()
        texture = 0.5 + 0.5 * torch.cos(radial * 11.0)
    texture = texture.unsqueeze(0)
    texture += 0.03 * torch.rand((1, image_size, image_size), generator=generator, dtype=torch.float32)
    return texture.clamp(0.0, 1.0)


def _make_subject(reference_mask: torch.Tensor, texture: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    background = 0.06 + 0.08 * torch.rand(reference_mask.shape, generator=generator, dtype=torch.float32)
    subject = background * (1.0 - reference_mask) + (0.18 + 0.74 * texture) * reference_mask
    subject += 0.02 * torch.rand(reference_mask.shape, generator=generator, dtype=torch.float32)
    return subject.clamp(0.0, 1.0)


def _make_target(guidance: torch.Tensor, texture: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    shifted_texture = 0.5 * texture + 0.5 * torch.roll(texture, shifts=1, dims=2)
    background = 0.04 + 0.06 * torch.rand(guidance.shape, generator=generator, dtype=torch.float32)
    foreground = 0.14 + 0.78 * shifted_texture
    target = background * (1.0 - guidance) + foreground * guidance
    target += 0.03 * torch.randn(guidance.shape, generator=generator, dtype=torch.float32)
    target += 0.02 * torch.rand(guidance.shape, generator=generator, dtype=torch.float32)
    return target.clamp(0.0, 1.0)


def _make_synthetic_subject_driven_data(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(cfg.seed))
    subjects: list[torch.Tensor] = []
    guidances: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for _ in range(int(cfg.num_samples)):
        reference_mask = _shape_mask(int(cfg.image_size), generator)
        guidance = _shape_mask(int(cfg.image_size), generator)
        texture = _subject_texture(int(cfg.image_size), generator)
        subject = _make_subject(reference_mask, texture, generator)
        target = _make_target(guidance, texture, generator)
        subjects.append(subject)
        guidances.append(guidance)
        targets.append(target)

    return torch.stack(subjects, dim=0), torch.stack(guidances, dim=0), torch.stack(targets, dim=0)


class SyntheticSubjectDrivenDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._subject, self._guidance, self._target = _make_synthetic_subject_driven_data(cfg)

    def __len__(self) -> int:
        return int(self._target.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._subject[i], self._guidance[i], self._target[i]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset: Dataset = SyntheticSubjectDrivenDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticSubjectDrivenDataset", "get_dataloaders"]
