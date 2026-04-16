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


class SyntheticLaneDetectionDataset(Dataset):
    """Render simple road scenes with a few lane centerlines and dense supervision."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        size = int(cfg.image_size)
        if size < 24:
            raise ValueError("image_size must be >= 24")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if int(cfg.min_lanes) < 1 or int(cfg.max_lanes) < int(cfg.min_lanes):
            raise ValueError("invalid lane count range")
        if float(cfg.lane_width) <= 0.5:
            raise ValueError("lane_width must be > 0.5")
        if float(cfg.noise_std) < 0.0:
            raise ValueError("noise_std must be >= 0")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _generator(self, idx: int) -> torch.Generator:
        seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 97 + 23
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
        road = 0.14 + 0.16 * (1.0 - y_norm) + 0.04 * torch.cos((x_norm - 0.5) * torch.pi)

        heatmap = torch.zeros((size, size), dtype=torch.float32)
        offset = torch.zeros((size, size), dtype=torch.float32)
        image = road.clone()

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
            curvature = float((torch.rand((), generator=gen).item() * 0.16) - 0.08) * size
            center_x = base_x + slope * (t - 0.5) + curvature * torch.square(t - 0.5)
            center_x = center_x.clamp(0.0, float(size - 1))
            center_grid = center_x.unsqueeze(1).expand(size, size)
            dist = torch.abs(xx - center_grid)
            gaussian = torch.exp(-0.5 * torch.square(dist / float(cfg.lane_width)))
            better = gaussian > heatmap
            heatmap = torch.maximum(heatmap, gaussian)
            offset = torch.where(better, center_grid / max(1.0, float(size - 1)), offset)

            lane_intensity = (0.25 + 0.70 * gaussian).clamp(0.0, 1.0)
            image = torch.maximum(image, lane_intensity)

            shoulder = torch.exp(-0.5 * torch.square(dist / float(cfg.lane_width * 2.2)))
            image = torch.maximum(image, 0.25 + 0.20 * shoulder)

        if float(cfg.noise_std) > 0.0:
            image = image + torch.randn((size, size), generator=gen, dtype=torch.float32) * float(
                cfg.noise_std
            )

        mask = (heatmap >= 0.15).to(torch.float32)
        image = image.clamp(0.0, 1.0).unsqueeze(0)
        target = {
            "heatmap": heatmap.unsqueeze(0),
            "offset": offset.unsqueeze(0),
            "mask": mask.unsqueeze(0),
            "lane_count": torch.tensor(float(num_lanes), dtype=torch.float32),
        }
        return image, target


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds = SyntheticLaneDetectionDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticLaneDetectionDataset", "get_dataloaders"]
