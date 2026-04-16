import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 256
    num_points: int = 96
    batch_size: int = 16
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    motion_scale: float = 0.35
    clutter_ratio: float = 0.2
    noise_std: float = 0.01


def _sample_object_shape(*, num_points: int, g: torch.Generator) -> torch.Tensor:
    radii = torch.tensor([0.35, 0.25, 0.2], dtype=torch.float32)
    local = (torch.rand((num_points, 3), generator=g, dtype=torch.float32) * 2.0 - 1.0) * radii
    return local


def _inject_clutter(
    cloud: torch.Tensor,
    *,
    clutter_ratio: float,
    g: torch.Generator,
) -> torch.Tensor:
    num_points = int(cloud.shape[0])
    clutter_points = int(round(float(clutter_ratio) * num_points))
    clutter_points = max(0, min(num_points, clutter_points))
    if clutter_points == 0:
        return cloud

    out = cloud.clone()
    idx = torch.randperm(num_points, generator=g)[:clutter_points]
    out[idx] = torch.rand((clutter_points, 3), generator=g, dtype=torch.float32) * 3.0 - 1.5
    return out


class ToyObjectTrackingDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.num_points) < 16:
            raise ValueError("num_points must be >= 16")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if float(cfg.motion_scale) < 0.0:
            raise ValueError("motion_scale must be >= 0")
        if not (0.0 <= float(cfg.clutter_ratio) <= 1.0):
            raise ValueError("clutter_ratio must be in [0, 1]")
        if float(cfg.noise_std) < 0.0:
            raise ValueError("noise_std must be >= 0")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_idx = int(idx)
        cfg = self.cfg
        g = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + sample_idx)
        num_points = int(cfg.num_points)

        center_prev = torch.rand((3,), generator=g, dtype=torch.float32) * 1.2 - 0.6
        phase = 2.0 * math.pi * ((sample_idx % 41) / 41.0)
        base_velocity = torch.tensor(
            [math.cos(phase), math.sin(phase), 0.5 * math.cos(0.5 * phase)],
            dtype=torch.float32,
        )
        jitter = 0.2 * torch.randn((3,), generator=g, dtype=torch.float32)
        velocity = (base_velocity + jitter) * float(cfg.motion_scale)
        center_curr = center_prev + velocity

        local_shape = _sample_object_shape(num_points=num_points, g=g)
        prev_cloud = center_prev.unsqueeze(0) + local_shape
        curr_cloud = center_curr.unsqueeze(0) + local_shape

        noise_scale = float(cfg.noise_std)
        if noise_scale > 0.0:
            prev_cloud = prev_cloud + torch.randn((num_points, 3), generator=g, dtype=torch.float32) * noise_scale
            curr_cloud = curr_cloud + torch.randn((num_points, 3), generator=g, dtype=torch.float32) * noise_scale

        prev_cloud = _inject_clutter(prev_cloud, clutter_ratio=float(cfg.clutter_ratio), g=g)
        curr_cloud = _inject_clutter(curr_cloud, clutter_ratio=float(cfg.clutter_ratio), g=g)

        prev_cloud = prev_cloud[torch.randperm(num_points, generator=g)]
        curr_cloud = curr_cloud[torch.randperm(num_points, generator=g)]

        target_state = torch.cat([center_curr, velocity], dim=0)
        return prev_cloud.to(torch.float32), curr_cloud.to(torch.float32), target_state.to(torch.float32)


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset = ToyObjectTrackingDataset(cfg)
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


__all__ = ["DataConfig", "ToyObjectTrackingDataset", "get_dataloaders"]
