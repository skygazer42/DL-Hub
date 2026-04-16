from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices
from tracks.pointcloud.toy_clouds import _sample_cube_surface, _sample_sphere


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 256
    num_points: int = 96
    batch_size: int = 16
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    anomaly_fraction: float = 0.35
    anomaly_scale: float = 0.55
    jitter_std: float = 0.01
    p_sphere: float = 0.5


def _sample_clean_cloud(cfg: DataConfig, g: torch.Generator) -> torch.Tensor:
    if bool(torch.rand((), generator=g).item() < float(cfg.p_sphere)):
        return _sample_sphere(num_points=int(cfg.num_points), g=g, noise_std=0.01)
    return _sample_cube_surface(num_points=int(cfg.num_points), g=g, noise_std=0.0)


class SyntheticPointCloudAnomalyDataset(Dataset):
    """Synthetic anomaly detection dataset with point-level labels."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.num_samples) <= 0:
            raise ValueError("num_samples must be > 0")
        if int(cfg.num_points) < 16:
            raise ValueError("num_points must be >= 16")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if not (0.0 <= float(cfg.anomaly_fraction) <= 1.0):
            raise ValueError("anomaly_fraction must be in [0, 1]")
        if float(cfg.anomaly_scale) < 0.0:
            raise ValueError("anomaly_scale must be >= 0")
        if float(cfg.jitter_std) < 0.0:
            raise ValueError("jitter_std must be >= 0")
        if not (0.0 <= float(cfg.p_sphere) <= 1.0):
            raise ValueError("p_sphere must be in [0, 1]")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        sample_idx = int(idx)
        g = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + sample_idx)

        clean = _sample_clean_cloud(cfg, g).to(torch.float32)
        points = clean.clone()
        point_labels = torch.zeros(int(cfg.num_points), dtype=torch.float32)

        is_anomaly = bool(torch.rand((), generator=g).item() < float(cfg.anomaly_fraction))
        if is_anomaly:
            num_anomaly = max(4, int(round(0.18 * float(cfg.num_points))))
            perm = torch.randperm(int(cfg.num_points), generator=g)
            anomaly_idx = perm[:num_anomaly]
            direction = torch.randn((3,), generator=g, dtype=torch.float32)
            direction = direction / direction.norm().clamp(min=1e-6)
            offset = direction * float(cfg.anomaly_scale)
            points[anomaly_idx] = points[anomaly_idx] + offset
            point_labels[anomaly_idx] = 1.0

        if float(cfg.jitter_std) > 0.0:
            points = points + torch.randn(points.shape, generator=g, dtype=torch.float32) * float(
                cfg.jitter_std
            )

        return points, {
            "reconstruction": clean,
            "point_labels": point_labels,
            "label": point_labels.max(),
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds = SyntheticPointCloudAnomalyDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticPointCloudAnomalyDataset", "get_dataloaders"]
