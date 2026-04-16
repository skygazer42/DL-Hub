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

    near_depth: float = 0.2
    far_depth: float = 0.9
    min_layers: int = 2
    max_layers: int = 4
    add_gradient_background: bool = True
    noise_std: float = 0.02


class SyntheticMonocularDepthDataset(Dataset):
    """Synthetic grayscale scenes with layered shapes and dense depth supervision."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        size = int(cfg.image_size)
        if size < 16:
            raise ValueError("image_size must be >= 16")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if not (0.0 <= float(cfg.near_depth) < float(cfg.far_depth) <= 1.0):
            raise ValueError("near_depth and far_depth must satisfy 0 <= near < far <= 1")
        if int(cfg.min_layers) < 1 or int(cfg.max_layers) < int(cfg.min_layers):
            raise ValueError("invalid layer range")
        if float(cfg.noise_std) < 0.0:
            raise ValueError("noise_std must be >= 0")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _generator(self, idx: int) -> torch.Generator:
        seed = int(self.cfg.seed) * 1_000_003 + int(idx) * 97 + 19
        return torch.Generator().manual_seed(seed)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        idx = int(idx)
        cfg = self.cfg
        size = int(cfg.image_size)
        gen = self._generator(idx)

        yy, xx = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing="ij",
        )

        x_norm = xx / max(1.0, float(size - 1))
        y_norm = yy / max(1.0, float(size - 1))

        bg = 0.18 + 0.22 * x_norm + 0.14 * y_norm if cfg.add_gradient_background else torch.full(
            (size, size), 0.25, dtype=torch.float32
        )
        depth = torch.full((size, size), float(cfg.far_depth), dtype=torch.float32)
        image = bg.clone()
        occlusion = torch.zeros((size, size), dtype=torch.float32)
        layer_ids = torch.zeros((size, size), dtype=torch.long)

        num_layers = int(
            torch.randint(
                int(cfg.min_layers),
                int(cfg.max_layers) + 1,
                (1,),
                generator=gen,
            ).item()
        )
        depth_values = torch.linspace(
            float(cfg.far_depth) - 0.08,
            float(cfg.near_depth),
            steps=num_layers,
            dtype=torch.float32,
        )

        min_extent = max(4, size // 6)
        max_extent = max(min_extent + 1, size // 2)

        for layer_idx in range(num_layers):
            is_rect = bool(torch.rand((), generator=gen).item() < 0.5)
            width = int(
                torch.randint(min_extent, max_extent + 1, (1,), generator=gen).item()
            )
            height = int(
                torch.randint(min_extent, max_extent + 1, (1,), generator=gen).item()
            )
            cx = int(torch.randint(0, size, (1,), generator=gen).item())
            cy = int(torch.randint(0, size, (1,), generator=gen).item())

            if is_rect:
                half_w = max(2, width // 2)
                half_h = max(2, height // 2)
                mask = (
                    (xx >= float(cx - half_w))
                    & (xx <= float(cx + half_w))
                    & (yy >= float(cy - half_h))
                    & (yy <= float(cy + half_h))
                )
            else:
                radius_x = max(3.0, float(width) * 0.5)
                radius_y = max(3.0, float(height) * 0.5)
                mask = (
                    ((xx - float(cx)) / radius_x) ** 2 + ((yy - float(cy)) / radius_y) ** 2
                ) <= 1.0

            if not bool(mask.any()):
                continue

            layer_depth = float(depth_values[layer_idx].item())
            tone = 0.85 - 0.55 * layer_depth
            stripe = 0.05 * torch.sin((xx + yy) / max(2.0, float(3 + layer_idx)))
            patch = (tone + stripe).clamp(0.0, 1.0)

            depth[mask] = layer_depth
            image[mask] = patch[mask]
            occlusion[mask] = 1.0
            layer_ids[mask] = layer_idx + 1

        if float(cfg.noise_std) > 0.0:
            image = image + torch.randn((size, size), generator=gen, dtype=torch.float32) * float(
                cfg.noise_std
            )

        image = image.clamp(0.0, 1.0).unsqueeze(0)
        target = {
            "depth": depth.unsqueeze(0),
            "occlusion": occlusion.unsqueeze(0),
            "layer_ids": layer_ids.unsqueeze(0),
        }
        return image, target


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    ds = SyntheticMonocularDepthDataset(cfg)
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


__all__ = ["DataConfig", "SyntheticMonocularDepthDataset", "get_dataloaders"]

