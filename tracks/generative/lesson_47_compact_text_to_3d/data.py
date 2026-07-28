from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 8
    num_workers: int = 0
    num_samples: int = 64
    seed: int = 0
    val_fraction: float = 0.2
    text_dim: int = 32
    voxel_size: int = 12


def _prompt_features(num_samples: int, text_dim: int, generator: torch.Generator) -> torch.Tensor:
    token_bank = torch.tensor(
        [
            [0.95, 0.15, 0.20, 0.10],
            [0.20, 0.85, 0.30, 0.10],
            [0.25, 0.35, 0.95, 0.15],
            [0.85, 0.85, 0.15, 0.20],
            [0.70, 0.20, 0.85, 0.65],
            [0.25, 0.80, 0.90, 0.75],
            [0.90, 0.55, 0.15, 0.70],
            [0.30, 0.90, 0.70, 0.85],
        ],
        dtype=torch.float32,
    )
    num_tokens = token_bank.shape[0]
    features = torch.zeros((int(num_samples), int(text_dim)), dtype=torch.float32)
    for idx in range(int(num_samples)):
        token_count = int(torch.randint(2, 5, (1,), generator=generator).item())
        token_ids = torch.randint(0, num_tokens, (token_count,), generator=generator)
        seq = token_bank.index_select(0, token_ids).flatten()
        usable = min(seq.numel(), int(text_dim))
        features[idx, :usable] = seq[:usable]
        features[idx] += 0.02 * torch.randn(int(text_dim), generator=generator)
    return features.clamp(0.0, 1.0)


def _density_from_text(text: torch.Tensor, voxel_size: int) -> torch.Tensor:
    batch = int(text.shape[0])
    device = text.device
    yy, xx, zz = torch.meshgrid(
        torch.linspace(-1.0, 1.0, int(voxel_size), dtype=torch.float32, device=device),
        torch.linspace(-1.0, 1.0, int(voxel_size), dtype=torch.float32, device=device),
        torch.linspace(-1.0, 1.0, int(voxel_size), dtype=torch.float32, device=device),
        indexing="ij",
    )
    centers = (text[:, :3] * 1.4) - 0.7
    widths = 0.20 + 0.45 * text[:, 3:4]
    outputs: list[torch.Tensor] = []
    for i in range(batch):
        cx, cy, cz = [float(v.item()) for v in centers[i]]
        width = float(widths[i, 0].item())
        dist = (xx - cx).pow(2) + (yy - cy).pow(2) + (zz - cz).pow(2)
        density = torch.exp(-dist / max(1e-6, 2.0 * width * width))
        outputs.append(density.unsqueeze(0))
    return torch.stack(outputs, dim=0)


def _mesh_tokens_from_text(text: torch.Tensor, voxel_size: int) -> torch.Tensor:
    batch = int(text.shape[0])
    theta = torch.linspace(0.0, 2.0 * torch.pi, int(voxel_size), dtype=torch.float32)
    mesh = torch.zeros((batch, int(voxel_size), 3), dtype=torch.float32)
    for i in range(batch):
        radius = 0.2 + 0.7 * float(text[i, 4].item())
        height = 0.2 + 0.7 * float(text[i, 5].item())
        wobble = 0.05 + 0.15 * float(text[i, 6].item())
        mesh[i, :, 0] = radius * torch.cos(theta) + wobble * torch.sin(theta * 2.0)
        mesh[i, :, 1] = radius * torch.sin(theta)
        mesh[i, :, 2] = torch.linspace(-height, height, int(voxel_size), dtype=torch.float32)
    return mesh


def _make_synthetic_text_to_3d_data(cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(cfg.seed))
    text = _prompt_features(int(cfg.num_samples), int(cfg.text_dim), generator)
    density = _density_from_text(text, int(cfg.voxel_size))
    mesh_tokens = _mesh_tokens_from_text(text, int(cfg.voxel_size))
    return text, density, mesh_tokens


class SyntheticTextTo3DDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._text, self._density, self._mesh_tokens = _make_synthetic_text_to_3d_data(cfg)

    def __len__(self) -> int:
        return int(self._text.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        i = int(idx)
        targets = {
            "density": self._density[i],
            "mesh_tokens": self._mesh_tokens[i],
        }
        return self._text[i], targets


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset: Dataset = SyntheticTextTo3DDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
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


__all__ = ["DataConfig", "SyntheticTextTo3DDataset", "get_dataloaders"]
