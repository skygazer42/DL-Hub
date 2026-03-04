from __future__ import annotations

from dataclasses import dataclass

import math

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


def _sample_cube(*, num_points: int, g: torch.Generator) -> torch.Tensor:
    return torch.rand((int(num_points), 3), generator=g, dtype=torch.float32) * 2.0 - 1.0


def _sample_sphere(*, num_points: int, g: torch.Generator) -> torch.Tensor:
    pts = torch.randn((int(num_points), 3), generator=g, dtype=torch.float32)
    return pts / pts.norm(dim=1, keepdim=True).clamp(min=1e-8)


def _rand_uniform(g: torch.Generator, low: float, high: float, shape: tuple[int, ...] = ()) -> torch.Tensor:
    return torch.rand(shape, generator=g, dtype=torch.float32) * (float(high) - float(low)) + float(low)


def _augment(points: torch.Tensor, *, g: torch.Generator, cfg) -> torch.Tensor:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape (N, 3), got {tuple(points.shape)}")

    x = points.clone()

    # Random rotation around z-axis.
    if float(cfg.rotate_deg) > 0:
        max_theta = math.pi * min(180.0, float(cfg.rotate_deg)) / 180.0
        theta = _rand_uniform(g, -max_theta, max_theta)
        c = torch.cos(theta)
        s = torch.sin(theta)
        r = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        x = x @ r.t()

    # Random scaling.
    if float(cfg.scale_low) != 1.0 or float(cfg.scale_high) != 1.0:
        scale = _rand_uniform(g, float(cfg.scale_low), float(cfg.scale_high))
        x = x * scale

    # Random translation.
    if float(cfg.translate_std) > 0:
        t = torch.randn((1, 3), generator=g, dtype=torch.float32) * float(cfg.translate_std)
        x = x + t

    # Jitter.
    if float(cfg.jitter_std) > 0:
        noise = torch.randn(x.shape, generator=g, dtype=torch.float32) * float(cfg.jitter_std)
        x = x + noise

    # Point dropout.
    drop_p = float(cfg.drop_p)
    if drop_p > 0:
        mask = torch.rand((x.shape[0],), generator=g, dtype=torch.float32) < drop_p
        if int(mask.sum().item()) > 0:
            x[mask] = x[0].clone()

    if bool(cfg.shuffle_points):
        perm = torch.randperm(x.shape[0], generator=g)
        x = x[perm]

    return x


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 4096
    num_points: int = 128
    batch_size: int = 128
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0

    p_sphere: float = 0.5

    rotate_deg: float = 180.0
    scale_low: float = 0.8
    scale_high: float = 1.2
    translate_std: float = 0.05
    jitter_std: float = 0.01
    drop_p: float = 0.1
    shuffle_points: bool = True


class ToySSLViewsDataset(Dataset):
    """Return two augmented views + label (label only for optional probing)."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.num_points) < 16:
            raise ValueError("num_points must be >= 16")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if not (0.0 <= float(cfg.p_sphere) <= 1.0):
            raise ValueError("p_sphere must be in [0, 1]")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(idx)
        cfg = self.cfg

        g0 = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + i)
        is_sphere = bool(torch.rand((), generator=g0).item() < float(cfg.p_sphere))
        label = torch.tensor(1 if is_sphere else 0, dtype=torch.long)

        if is_sphere:
            base = _sample_sphere(num_points=int(cfg.num_points), g=g0)
        else:
            base = _sample_cube(num_points=int(cfg.num_points), g=g0)

        g1 = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + i * 17 + 1)
        g2 = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + i * 17 + 2)
        v1 = _augment(base, g=g1, cfg=cfg)
        v2 = _augment(base, g=g2, cfg=cfg)
        return v1, v2, label


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds = ToySSLViewsDataset(cfg)
    train_idx, val_idx = train_val_split_indices(n=len(ds), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed))

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    def _collate(batch):
        v1 = torch.stack([b[0] for b in batch], dim=0)
        v2 = torch.stack([b[1] for b in batch], dim=0)
        y = torch.stack([b[2] for b in batch], dim=0)
        return v1, v2, y

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
        drop_last=False,
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "ToySSLViewsDataset", "get_dataloaders"]

