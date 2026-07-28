from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices
from tracks.pointcloud.synthetic_clouds import _sample_cube_surface, _sample_sphere


def _render_target_splats(points: torch.Tensor, *, image_size: int, sigma: float) -> torch.Tensor:
    if points.ndim != 2 or points.size(-1) != 3:
        raise ValueError("expected points shaped [num_points, 3]")

    xy = points[:, :2].clamp(-1.0, 1.0)
    centers_x = xy[:, 0].view(-1, 1, 1)
    centers_y = xy[:, 1].view(-1, 1, 1)

    grid = torch.linspace(-1.0, 1.0, int(image_size), dtype=torch.float32)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)

    var = float(sigma) * float(sigma)
    sq_dist = (xx - centers_x).square() + (yy - centers_y).square()
    gaussians = torch.exp(-0.5 * sq_dist / max(var, 1e-6))
    image = gaussians.mean(dim=0, keepdim=True)
    return image.to(torch.float32)


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 256
    num_points: int = 96
    image_size: int = 24
    batch_size: int = 16
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    noise_std: float = 0.03
    p_sphere: float = 0.5
    splat_sigma: float = 0.08


class SyntheticGaussianSplattingDataset(Dataset):
    """Synthetic point-cloud-to-image supervision for synthetic Gaussian splatting."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.num_points) < 16:
            raise ValueError("num_points must be >= 16")
        if int(cfg.image_size) < 8:
            raise ValueError("image_size must be >= 8")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if float(cfg.noise_std) < 0.0:
            raise ValueError("noise_std must be >= 0")
        if not (0.0 <= float(cfg.p_sphere) <= 1.0):
            raise ValueError("p_sphere must be in [0, 1]")
        if float(cfg.splat_sigma) <= 0.0:
            raise ValueError("splat_sigma must be > 0")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        sample_seed = int(cfg.seed) * 1_000_003 + int(idx)
        g = torch.Generator().manual_seed(sample_seed)
        num_points = int(cfg.num_points)

        is_sphere = bool(torch.rand((), generator=g).item() < float(cfg.p_sphere))
        if is_sphere:
            clean = _sample_sphere(num_points=num_points, g=g, noise_std=0.0)
        else:
            clean = _sample_cube_surface(num_points=num_points, g=g, noise_std=0.0)

        observed = clean + torch.randn((num_points, 3), generator=g, dtype=torch.float32) * float(
            cfg.noise_std
        )
        observed = observed.clamp(-1.25, 1.25)

        target = _render_target_splats(
            clean,
            image_size=int(cfg.image_size),
            sigma=float(cfg.splat_sigma),
        )
        return observed.to(torch.float32), target.to(torch.float32)


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset = SyntheticGaussianSplattingDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticGaussianSplattingDataset", "get_dataloaders"]
