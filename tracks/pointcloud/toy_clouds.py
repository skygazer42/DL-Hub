
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


def _sample_cube(*, num_points: int, g: torch.Generator) -> torch.Tensor:
    # Uniform points inside a cube.
    pts = torch.rand((num_points, 3), generator=g, dtype=torch.float32) * 2.0 - 1.0
    return pts


def _sample_cube_surface(*, num_points: int, g: torch.Generator, noise_std: float = 0.0) -> torch.Tensor:
    """Sample points on the surface of a unit cube [-1,1]^3."""

    n = int(num_points)
    face = torch.randint(0, 6, (n,), generator=g, dtype=torch.long)
    pts = torch.rand((n, 3), generator=g, dtype=torch.float32) * 2.0 - 1.0

    # faces: 0/1 -> x=-1/+1, 2/3 -> y=-1/+1, 4/5 -> z=-1/+1
    axis = face // 2  # 0,1,2
    sign = (face % 2) * 2 - 1  # -1 or +1
    for ax in (0, 1, 2):
        mask = axis == ax
        if int(mask.sum().item()) > 0:
            pts[mask, ax] = sign[mask].to(torch.float32)

    ns = float(noise_std)
    if ns > 0.0:
        noise = torch.randn((n, 3), generator=g, dtype=torch.float32) * ns
        pts = pts + noise
    return pts


def _sample_sphere(*, num_points: int, g: torch.Generator, noise_std: float = 0.02) -> torch.Tensor:
    # Sample points near the unit sphere surface.
    pts = torch.randn((num_points, 3), generator=g, dtype=torch.float32)
    pts = pts / pts.norm(dim=1, keepdim=True).clamp(min=1e-8)
    noise = torch.randn(pts.shape, generator=g, dtype=torch.float32)
    pts = pts + noise * float(noise_std)
    return pts


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 2048
    num_points: int = 128
    batch_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


class ToyPointCloudDataset(Dataset):
    """Cube vs Sphere point cloud classification dataset (fully synthetic)."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        num_samples = int(cfg.num_samples)
        num_points = int(cfg.num_points)

        g = torch.Generator().manual_seed(int(cfg.seed))
        labels = torch.randint(low=0, high=2, size=(num_samples,), generator=g, dtype=torch.long)

        clouds = torch.empty((num_samples, num_points, 3), dtype=torch.float32)
        for i in range(num_samples):
            if int(labels[i].item()) == 0:
                clouds[i] = _sample_cube(num_points=num_points, g=g)
            else:
                clouds[i] = _sample_sphere(num_points=num_points, g=g)

        self.clouds = clouds
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(idx)
        return self.clouds[i], self.labels[i]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds = ToyPointCloudDataset(cfg)
    train_idx, val_idx = train_val_split_indices(n=len(ds), val_fraction=cfg.val_fraction, seed=cfg.seed)

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    return train_loader, val_loader


@dataclass(frozen=True)
class PartSegDataConfig:
    num_samples: int = 2048
    num_points: int = 256
    batch_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0

    noise_std: float = 0.02
    offset: float = 1.0  # separation between the two parts (along x-axis)
    shuffle_points: bool = True


class ToyPartSegDataset(Dataset):
    """Two-part point cloud segmentation: cube vs sphere in one sample.

    Output:
    - points: (N, 3)
    - labels: (N,) with classes {0=cube, 1=sphere}
    """

    def __init__(self, cfg: PartSegDataConfig) -> None:
        self.cfg = cfg
        if int(cfg.num_points) < 16:
            raise ValueError("num_points must be >= 16")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(idx)
        cfg = self.cfg
        n = int(cfg.num_points)
        n0 = n // 2
        n1 = n - n0

        # Deterministic per-sample RNG (stable across dataloader workers).
        g = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + i)

        cube = _sample_cube(num_points=n0, g=g)
        sphere = _sample_sphere(num_points=n1, g=g, noise_std=float(cfg.noise_std))

        off = float(cfg.offset) * 0.5
        cube = cube + torch.tensor([-off, 0.0, 0.0], dtype=torch.float32)
        sphere = sphere + torch.tensor([off, 0.0, 0.0], dtype=torch.float32)

        points = torch.cat([cube, sphere], dim=0)  # (N, 3)
        labels = torch.cat([torch.zeros(n0, dtype=torch.long), torch.ones(n1, dtype=torch.long)], dim=0)  # (N,)

        if bool(cfg.shuffle_points):
            perm = torch.randperm(n, generator=g)
            points = points[perm]
            labels = labels[perm]

        return points, labels


def get_partseg_dataloaders(cfg: PartSegDataConfig) -> tuple[DataLoader, DataLoader]:
    ds = ToyPartSegDataset(cfg)
    train_idx, val_idx = train_val_split_indices(n=len(ds), val_fraction=cfg.val_fraction, seed=cfg.seed)

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    return train_loader, val_loader


@dataclass(frozen=True)
class ReconDataConfig:
    num_samples: int = 2048
    num_points: int = 128
    batch_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0

    noise_std: float = 0.03
    sphere_surface_noise: float = 0.0
    cube_surface_noise: float = 0.0
    p_sphere: float = 0.5
    shuffle_points: bool = True


class ToyReconDataset(Dataset):
    """Toy point cloud reconstruction: noisy -> clean (set-to-set).

    Returns:
        noisy_points: (N, 3)
        clean_points: (N, 3)
    """

    def __init__(self, cfg: ReconDataConfig) -> None:
        self.cfg = cfg
        if int(cfg.num_points) < 16:
            raise ValueError("num_points must be >= 16")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if not (0.0 <= float(cfg.p_sphere) <= 1.0):
            raise ValueError("p_sphere must be in [0, 1]")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(idx)
        cfg = self.cfg
        n = int(cfg.num_points)

        g = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + i)
        is_sphere = bool(torch.rand((), generator=g).item() < float(cfg.p_sphere))

        if is_sphere:
            clean = _sample_sphere(num_points=n, g=g, noise_std=float(cfg.sphere_surface_noise))
        else:
            clean = _sample_cube_surface(num_points=n, g=g, noise_std=float(cfg.cube_surface_noise))

        noisy = clean + torch.randn((n, 3), generator=g, dtype=torch.float32) * float(cfg.noise_std)

        if bool(cfg.shuffle_points):
            perm = torch.randperm(n, generator=g)
            clean = clean[perm]
            noisy = noisy[perm]

        return noisy, clean


def get_recon_dataloaders(cfg: ReconDataConfig) -> tuple[DataLoader, DataLoader]:
    ds = ToyReconDataset(cfg)
    train_idx, val_idx = train_val_split_indices(n=len(ds), val_fraction=cfg.val_fraction, seed=cfg.seed)

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        drop_last=False,
    )
    return train_loader, val_loader


__all__ = [
    "DataConfig",
    "PartSegDataConfig",
    "ReconDataConfig",
    "ToyReconDataset",
    "ToyPartSegDataset",
    "ToyPointCloudDataset",
    "get_dataloaders",
    "get_partseg_dataloaders",
    "get_recon_dataloaders",
]
