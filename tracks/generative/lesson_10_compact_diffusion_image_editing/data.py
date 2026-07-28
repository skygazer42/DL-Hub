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


class SyntheticImageEditDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._source, self._target, self._mask, self._control = _make_synthetic_edit_data(cfg)

    def __len__(self) -> int:
        return int(self._source.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self._source[i], self._target[i], self._mask[i], self._control[i]


def _paint_source(image_size: int, generator: torch.Generator) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, image_size, dtype=torch.float32),
        indexing="ij",
    )
    image = torch.zeros((1, image_size, image_size), dtype=torch.float32)
    mode = int(torch.randint(low=0, high=3, size=(1,), generator=generator).item())

    if mode == 0:
        cx = float(torch.empty((1,)).uniform_(-0.6, 0.6, generator=generator).item())
        cy = float(torch.empty((1,)).uniform_(-0.6, 0.6, generator=generator).item())
        radius = float(torch.empty((1,)).uniform_(0.2, 0.5, generator=generator).item())
        fg = (xx - cx).pow(2) + (yy - cy).pow(2) <= radius**2
        image[0, fg] = 0.8
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

    noise = 0.05 * torch.rand(image.shape, generator=generator)
    return torch.clamp(image + noise, 0.0, 1.0)


def _edit_patch_and_mask(image_size: int, generator: torch.Generator) -> torch.Tensor:
    mask = torch.zeros((1, image_size, image_size), dtype=torch.float32)
    h = int(torch.randint(max(4, image_size // 5), max(5, image_size // 2), (1,), generator=generator).item())
    w = int(torch.randint(max(4, image_size // 5), max(5, image_size // 2), (1,), generator=generator).item())
    top = int(torch.randint(0, image_size - h + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, image_size - w + 1, (1,), generator=generator).item())
    mask[:, top : top + h, left : left + w] = 1.0
    return mask


def _apply_edit(source: torch.Tensor, mask: torch.Tensor, control_token: int) -> torch.Tensor:
    if int(control_token) == 0:
        edited_region = torch.full_like(source, 0.95)
    else:
        edited_region = torch.zeros_like(source)
    target = source * (1.0 - mask) + edited_region * mask
    return target.clamp(0.0, 1.0)


def _make_synthetic_edit_data(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(int(cfg.seed))
    source_images: list[torch.Tensor] = []
    target_images: list[torch.Tensor] = []
    edit_masks: list[torch.Tensor] = []
    control_tokens: list[torch.Tensor] = []

    for _ in range(int(cfg.num_samples)):
        source = _paint_source(int(cfg.image_size), g)
        mask = _edit_patch_and_mask(int(cfg.image_size), g)
        control = int(torch.randint(low=0, high=2, size=(1,), generator=g).item())
        target = _apply_edit(source, mask, control)

        source_images.append(source)
        target_images.append(target)
        edit_masks.append(mask)
        control_tokens.append(torch.tensor(control, dtype=torch.long))

    return (
        torch.stack(source_images, dim=0),
        torch.stack(target_images, dim=0),
        torch.stack(edit_masks, dim=0),
        torch.stack(control_tokens, dim=0),
    )


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds: Dataset = SyntheticImageEditDataset(cfg)
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
