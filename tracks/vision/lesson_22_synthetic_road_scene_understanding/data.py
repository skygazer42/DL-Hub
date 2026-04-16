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

    num_lane_slots: int = 3
    num_object_types: int = 3
    noise_std: float = 0.01


class SyntheticRoadSceneDataset(Dataset):
    """Render road scenes with lane slots, object presence, and scene-level labels."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        size = int(cfg.image_size)
        if size < 24:
            raise ValueError("image_size must be >= 24")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if int(cfg.num_lane_slots) != 3:
            raise ValueError("num_lane_slots must be 3 for this toy lesson")
        if int(cfg.num_object_types) != 3:
            raise ValueError("num_object_types must be 3 for this toy lesson")
        if float(cfg.noise_std) < 0.0:
            raise ValueError("noise_std must be >= 0")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _generator(self, idx: int) -> torch.Generator:
        seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 131 + 37
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

        image = 0.08 + 0.16 * (1.0 - y_norm)
        road_mask = (x_norm > 0.12) & (x_norm < 0.88)
        image = image + road_mask.to(torch.float32) * 0.18

        scene_label = int(torch.randint(0, 4, (1,), generator=gen).item())
        lane_targets = torch.zeros((3,), dtype=torch.float32)
        object_targets = torch.zeros((3,), dtype=torch.float32)

        lane_layouts = (
            torch.tensor([1.0, 1.0, 1.0]),
            torch.tensor([1.0, 1.0, 0.0]),
            torch.tensor([0.0, 1.0, 1.0]),
            torch.tensor([1.0, 1.0, 1.0]),
        )
        lane_targets.copy_(lane_layouts[scene_label])

        lane_centers = torch.tensor([size * 0.28, size * 0.50, size * 0.72], dtype=torch.float32)
        curvature = {0: 0.0, 1: -0.10, 2: 0.10, 3: 0.0}[scene_label] * size
        t = 1.0 - y_norm[:, 0]
        lane_width = max(1.8, float(size) / 24.0)
        for lane_idx, lane_on in enumerate(lane_targets):
            if lane_on.item() < 0.5:
                continue
            base_x = lane_centers[lane_idx]
            lane_curve = base_x + curvature * torch.square(t - 0.5)
            lane_curve = lane_curve.clamp(0.0, float(size - 1))
            center_grid = lane_curve.unsqueeze(1).expand(size, size)
            dist = torch.abs(xx - center_grid)
            gaussian = torch.exp(-0.5 * torch.square(dist / lane_width))
            image = torch.maximum(image, 0.30 + 0.55 * gaussian)

        if scene_label == 3:
            stripe_positions = torch.linspace(size * 0.60, size * 0.82, steps=5)
            for stripe_y in stripe_positions:
                stripe = (torch.abs(yy - stripe_y) <= max(1.0, size / 48.0)) & road_mask
                image = torch.where(stripe, torch.full_like(image, 0.95), image)

        object_prob_table = (
            (0.75, 0.10, 0.15),  # car-heavy straight scene
            (0.55, 0.10, 0.35),  # left merge with bicycles
            (0.55, 0.10, 0.35),  # right merge with bicycles
            (0.25, 0.75, 0.10),  # crosswalk with pedestrians
        )
        for obj_idx, prob in enumerate(object_prob_table[scene_label]):
            if torch.rand((), generator=gen).item() < prob:
                object_targets[obj_idx] = 1.0

        if float(object_targets.sum().item()) == 0.0:
            fallback = int(torch.randint(0, 3, (1,), generator=gen).item())
            object_targets[fallback] = 1.0

        road_left = int(size * 0.18)
        road_right = int(size * 0.82)
        for obj_idx, present in enumerate(object_targets):
            if present.item() < 0.5:
                continue
            center_x = int(torch.randint(road_left, road_right, (1,), generator=gen).item())
            center_y = int(torch.randint(int(size * 0.42), int(size * 0.80), (1,), generator=gen).item())
            if obj_idx == 0:
                half_h = max(2, size // 14)
                half_w = max(3, size // 12)
                mask = (torch.abs(xx - center_x) <= half_w) & (torch.abs(yy - center_y) <= half_h)
                image = torch.where(mask, torch.full_like(image, 0.78), image)
            elif obj_idx == 1:
                radius = max(2.0, size / 20.0)
                mask = torch.square(xx - center_x) + torch.square(yy - center_y) <= radius**2
                image = torch.where(mask, torch.full_like(image, 0.88), image)
            else:
                radius = max(2.0, size / 18.0)
                body = torch.square(xx - center_x) + torch.square(yy - center_y) <= radius**2
                wheel_y = center_y + radius
                wheel_left = torch.square(xx - (center_x - radius)) + torch.square(yy - wheel_y) <= (
                    max(1.5, radius * 0.55) ** 2
                )
                wheel_right = torch.square(xx - (center_x + radius)) + torch.square(yy - wheel_y) <= (
                    max(1.5, radius * 0.55) ** 2
                )
                image = torch.where(body | wheel_left | wheel_right, torch.full_like(image, 0.70), image)

        if float(cfg.noise_std) > 0.0:
            noise = torch.randn((size, size), generator=gen, dtype=torch.float32) * float(cfg.noise_std)
            image = image + noise

        image = image.clamp(0.0, 1.0).unsqueeze(0)
        target = {
            "lane_targets": lane_targets,
            "object_targets": object_targets,
            "scene_label": torch.tensor(scene_label, dtype=torch.int64),
        }
        return image, target


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds = SyntheticRoadSceneDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticRoadSceneDataset", "get_dataloaders"]
