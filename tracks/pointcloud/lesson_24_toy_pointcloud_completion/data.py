from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices
from tracks.pointcloud.toy_clouds import _sample_cube_surface, _sample_sphere


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 2048
    num_points: int = 128
    batch_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    visible_fraction: float = 0.55
    p_sphere: float = 0.5
    partial_noise_std: float = 0.02
    sphere_surface_noise: float = 0.0
    cube_surface_noise: float = 0.0
    shuffle_points: bool = True


def _make_partial_observation(
    clean: torch.Tensor,
    *,
    visible_fraction: float,
    g: torch.Generator,
    partial_noise_std: float,
) -> torch.Tensor:
    num_points = int(clean.shape[0])
    keep = max(4, min(num_points, int(round(float(visible_fraction) * num_points))))

    direction = torch.randn((3,), generator=g, dtype=torch.float32)
    direction = direction / direction.norm().clamp(min=1e-6)
    scores = clean @ direction
    visible_idx = torch.topk(scores, k=keep, largest=True, sorted=False).indices
    visible = clean[visible_idx]

    resample_idx = torch.randint(0, keep, (num_points,), generator=g, dtype=torch.long)
    partial = visible[resample_idx]
    partial = partial + torch.randn((num_points, 3), generator=g, dtype=torch.float32) * float(
        partial_noise_std
    )
    return partial


class ToyPointCloudCompletionDataset(Dataset):
    """Toy partial-to-complete pointcloud completion pairs."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.num_points) < 16:
            raise ValueError("num_points must be >= 16")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if not (0.0 < float(cfg.visible_fraction) <= 1.0):
            raise ValueError("visible_fraction must be in (0, 1]")
        if not (0.0 <= float(cfg.p_sphere) <= 1.0):
            raise ValueError("p_sphere must be in [0, 1]")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        sample_seed = int(cfg.seed) * 1_000_003 + int(idx)
        g = torch.Generator().manual_seed(sample_seed)
        num_points = int(cfg.num_points)

        is_sphere = bool(torch.rand((), generator=g).item() < float(cfg.p_sphere))
        if is_sphere:
            clean = _sample_sphere(num_points=num_points, g=g, noise_std=float(cfg.sphere_surface_noise))
        else:
            clean = _sample_cube_surface(num_points=num_points, g=g, noise_std=float(cfg.cube_surface_noise))

        partial = _make_partial_observation(
            clean,
            visible_fraction=float(cfg.visible_fraction),
            g=g,
            partial_noise_std=float(cfg.partial_noise_std),
        )

        if bool(cfg.shuffle_points):
            clean = clean[torch.randperm(num_points, generator=g)]
            partial = partial[torch.randperm(num_points, generator=g)]

        return partial.to(torch.float32), clean.to(torch.float32)


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset = ToyPointCloudCompletionDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
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


__all__ = ["DataConfig", "ToyPointCloudCompletionDataset", "get_dataloaders"]
