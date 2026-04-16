from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 16
    image_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0

    min_lanes: int = 2
    max_lanes: int = 4
    lane_width: float = 2.5
    noise_std: float = 0.01


class SyntheticLaneTopologyDataset(Dataset):
    """Render simple road scenes with lane-wise heatmaps and adjacency labels."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        size = int(cfg.image_size)
        if size < 24:
            raise ValueError("image_size must be >= 24")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if int(cfg.min_lanes) < 2 or int(cfg.max_lanes) < int(cfg.min_lanes):
            raise ValueError("invalid lane count range")
        if float(cfg.lane_width) <= 0.5:
            raise ValueError("lane_width must be > 0.5")
        if float(cfg.noise_std) < 0.0:
            raise ValueError("noise_std must be >= 0")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _generator(self, idx: int) -> torch.Generator:
        seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 97 + 29
        return torch.Generator().manual_seed(seed)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        size = int(cfg.image_size)
        gen = self._generator(int(idx))

        yy, xx = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing="ij",
        )
        x_norm = xx / max(1.0, float(size - 1))
        y_norm = yy / max(1.0, float(size - 1))

        image = 0.12 + 0.18 * (1.0 - y_norm) + 0.05 * torch.cos((x_norm - 0.5) * torch.pi)
        lane_heatmaps = torch.zeros((int(cfg.max_lanes), size, size), dtype=torch.float32)
        lane_presence = torch.zeros((int(cfg.max_lanes),), dtype=torch.float32)
        adjacency = torch.zeros((int(cfg.max_lanes), int(cfg.max_lanes)), dtype=torch.float32)

        num_lanes = int(
            torch.randint(
                int(cfg.min_lanes),
                int(cfg.max_lanes) + 1,
                (1,),
                generator=gen,
            ).item()
        )

        anchors = torch.linspace(size * 0.22, size * 0.78, steps=num_lanes, dtype=torch.float32)
        anchor_jitter = torch.randn((num_lanes,), generator=gen, dtype=torch.float32) * (size * 0.03)
        base_positions = (anchors + anchor_jitter).clamp(0.1 * size, 0.9 * size)
        t = 1.0 - y_norm[:, 0]

        for lane_idx in range(num_lanes):
            base_x = float(base_positions[lane_idx].item())
            slope = float((torch.rand((), generator=gen).item() * 0.24) - 0.12) * size
            curvature = float((torch.rand((), generator=gen).item() * 0.14) - 0.07) * size
            center_x = base_x + slope * (t - 0.5) + curvature * torch.square(t - 0.5)
            center_x = center_x.clamp(0.0, float(size - 1))
            center_grid = center_x.unsqueeze(1).expand(size, size)
            dist = torch.abs(xx - center_grid)
            gaussian = torch.exp(-0.5 * torch.square(dist / float(cfg.lane_width)))

            lane_heatmaps[lane_idx] = gaussian
            lane_presence[lane_idx] = 1.0
            image = torch.maximum(image, (0.22 + 0.72 * gaussian).clamp(0.0, 1.0))

        for lane_idx in range(num_lanes - 1):
            adjacency[lane_idx, lane_idx + 1] = 1.0
            adjacency[lane_idx + 1, lane_idx] = 1.0

        if float(cfg.noise_std) > 0.0:
            image = image + torch.randn((size, size), generator=gen, dtype=torch.float32) * float(
                cfg.noise_std
            )

        image = image.clamp(0.0, 1.0).unsqueeze(0)
        target = {
            "lane_heatmaps": lane_heatmaps,
            "adjacency": adjacency,
            "lane_presence": lane_presence,
        }
        return image, target


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds = SyntheticLaneTopologyDataset(cfg)
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
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "SyntheticLaneTopologyDataset", "get_dataloaders"]
