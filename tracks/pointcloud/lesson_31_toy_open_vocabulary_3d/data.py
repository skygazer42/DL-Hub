from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices
from tracks.pointcloud.toy_clouds import _sample_cube_surface, _sample_sphere

_COLOR_WORDS = (
    ("red", "crimson"),
    ("blue", "azure"),
    ("green", "emerald"),
)
_SHAPE_WORDS = (
    ("cube", "box"),
    ("sphere", "ball"),
    ("cylinder", "column"),
)
_BASE_VOCAB = (
    "<pad>",
    "locate",
    "the",
    "target",
    "red",
    "crimson",
    "blue",
    "azure",
    "green",
    "emerald",
    "cube",
    "box",
    "sphere",
    "ball",
    "cylinder",
    "column",
)


def _sample_cylinder_surface(*, num_points: int, g: torch.Generator) -> torch.Tensor:
    theta = torch.rand((num_points,), generator=g, dtype=torch.float32) * (2.0 * torch.pi)
    z = torch.rand((num_points,), generator=g, dtype=torch.float32) * 2.0 - 1.0
    x = torch.cos(theta)
    y = torch.sin(theta)
    return torch.stack((x, y, z), dim=-1)


def _sample_shape_points(*, shape_id: int, num_points: int, g: torch.Generator) -> torch.Tensor:
    if int(shape_id) == 0:
        return _sample_cube_surface(num_points=num_points, g=g, noise_std=0.0)
    if int(shape_id) == 1:
        return _sample_sphere(num_points=num_points, g=g, noise_std=0.005)
    return _sample_cylinder_surface(num_points=num_points, g=g)


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 256
    num_points: int = 96
    batch_size: int = 16
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    max_text_length: int = 8
    jitter_std: float = 0.01


class ToyOpenVocabulary3DDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        if int(cfg.num_points) < 16:
            raise ValueError("num_points must be >= 16")
        if int(cfg.num_points) % 2 != 0:
            raise ValueError("num_points must be even")
        if not (0.0 < float(cfg.val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        if int(cfg.max_text_length) < 4:
            raise ValueError("max_text_length must be >= 4")
        self._token_to_id = {token: i for i, token in enumerate(_BASE_VOCAB)}
        self.pad_id = int(self._token_to_id["<pad>"])
        self.vocab_size = int(len(_BASE_VOCAB))

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def _encode_query(self, words: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = int(self.cfg.max_text_length)
        ids = torch.full((max_len,), fill_value=self.pad_id, dtype=torch.long)
        mask = torch.zeros((max_len,), dtype=torch.float32)
        clipped = words[:max_len]
        for pos, token in enumerate(clipped):
            ids[pos] = int(self._token_to_id[token])
            mask[pos] = 1.0
        return ids, mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_idx = int(idx)
        cfg = self.cfg
        num_points = int(cfg.num_points)
        obj_points = num_points // 2
        g = torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + sample_idx)

        shape_a = int(torch.randint(0, 3, (1,), generator=g).item())
        shape_b = int(torch.randint(0, 3, (1,), generator=g).item())
        color_a = int(torch.randint(0, 3, (1,), generator=g).item())
        color_b = int(torch.randint(0, 3, (1,), generator=g).item())

        points_a = _sample_shape_points(shape_id=shape_a, num_points=obj_points, g=g)
        points_b = _sample_shape_points(shape_id=shape_b, num_points=obj_points, g=g)
        points_a = points_a + torch.tensor([-1.1, 0.0, 0.0], dtype=torch.float32)
        points_b = points_b + torch.tensor([1.1, 0.0, 0.0], dtype=torch.float32)

        points = torch.cat((points_a, points_b), dim=0)
        point_mask = torch.cat(
            (torch.ones(obj_points, dtype=torch.float32), torch.zeros(obj_points, dtype=torch.float32)),
            dim=0,
        )
        class_label = torch.tensor(shape_a, dtype=torch.long)
        color_word = _COLOR_WORDS[color_a][sample_idx % len(_COLOR_WORDS[color_a])]
        shape_word = _SHAPE_WORDS[shape_a][sample_idx % len(_SHAPE_WORDS[shape_a])]
        query_ids, query_mask = self._encode_query(["locate", "the", color_word, shape_word, "target"])

        perm = torch.randperm(num_points, generator=g)
        points = points[perm]
        point_mask = point_mask[perm]
        points = points + torch.randn((num_points, 3), generator=g, dtype=torch.float32) * float(cfg.jitter_std)

        return (
            points.to(torch.float32),
            query_ids,
            query_mask,
            class_label,
            point_mask.to(torch.float32),
        )


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset = ToyOpenVocabulary3DDataset(cfg)
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


__all__ = ["DataConfig", "ToyOpenVocabulary3DDataset", "get_dataloaders"]
