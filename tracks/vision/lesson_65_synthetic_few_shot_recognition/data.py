from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


_CLASS_NAMES = ("ring", "vertical_bar", "horizontal_bar", "cross", "x_shape", "diamond")


@dataclass(frozen=True)
class DataConfig:
    num_episodes: int = 512
    batch_size: int = 8
    num_ways: int = 3
    shots: int = 2
    queries_per_class: int = 3
    image_size: int = 48
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    in_channels: int = 1
    noise_std: float = 0.035


@dataclass(frozen=True)
class SampleSpec:
    class_name: str
    label: int
    seed: int


@dataclass(frozen=True)
class EpisodeSpec:
    support_samples: tuple[SampleSpec, ...]
    query_samples: tuple[SampleSpec, ...]
    class_names: tuple[str, ...]


def _validate_config(cfg: DataConfig) -> None:
    if int(cfg.image_size) < 32:
        raise ValueError("image_size must be >= 32")
    if int(cfg.in_channels) != 1:
        raise ValueError("This lesson expects grayscale inputs.")
    if int(cfg.num_episodes) < 2:
        raise ValueError("num_episodes must be >= 2")
    if int(cfg.num_ways) < 2:
        raise ValueError("num_ways must be >= 2")
    if int(cfg.num_ways) > len(_CLASS_NAMES):
        raise ValueError(f"num_ways must be <= {len(_CLASS_NAMES)}")
    if int(cfg.shots) < 1:
        raise ValueError("shots must be >= 1")
    if int(cfg.queries_per_class) < 1:
        raise ValueError("queries_per_class must be >= 1")
    if not (0.0 < float(cfg.val_fraction) < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")
    if float(cfg.noise_std) < 0.0:
        raise ValueError("noise_std must be >= 0")


def _shuffle_samples(samples: list[SampleSpec], rng: np.random.Generator) -> tuple[SampleSpec, ...]:
    order = rng.permutation(len(samples))
    return tuple(samples[int(idx)] for idx in order)


def _make_episode_specs(cfg: DataConfig) -> list[EpisodeSpec]:
    _validate_config(cfg)
    rng = np.random.default_rng(int(cfg.seed))
    episodes: list[EpisodeSpec] = []

    for _episode_idx in range(int(cfg.num_episodes)):
        chosen = rng.choice(len(_CLASS_NAMES), size=int(cfg.num_ways), replace=False)
        support_samples: list[SampleSpec] = []
        query_samples: list[SampleSpec] = []
        class_names: list[str] = []

        for local_label, global_idx in enumerate(chosen):
            class_name = _CLASS_NAMES[int(global_idx)]
            class_names.append(class_name)
            for _shot in range(int(cfg.shots)):
                support_samples.append(
                    SampleSpec(
                        class_name=class_name,
                        label=local_label,
                        seed=int(rng.integers(0, 2_147_483_647)),
                    )
                )
            for _query in range(int(cfg.queries_per_class)):
                query_samples.append(
                    SampleSpec(
                        class_name=class_name,
                        label=local_label,
                        seed=int(rng.integers(0, 2_147_483_647)),
                    )
                )

        episodes.append(
            EpisodeSpec(
                support_samples=_shuffle_samples(support_samples, rng),
                query_samples=_shuffle_samples(query_samples, rng),
                class_names=tuple(class_names),
            )
        )
    return episodes


def _render_shape(class_name: str, seed: int, *, image_size: int, noise_std: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    size = int(image_size)
    coords = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")

    shift_x = float(rng.uniform(-0.18, 0.18))
    shift_y = float(rng.uniform(-0.18, 0.18))
    scale = float(rng.uniform(0.72, 0.96))
    sigma = float(rng.uniform(0.05, 0.09))
    dx = (xx - shift_x) / scale
    dy = (yy - shift_y) / scale

    if class_name == "ring":
        radius = float(rng.uniform(0.45, 0.62))
        mask = np.exp(-((np.sqrt(dx * dx + dy * dy) - radius) ** 2) / (2.0 * sigma * sigma))
    elif class_name == "vertical_bar":
        length = float(rng.uniform(0.8, 1.15))
        mask = np.exp(-(dx * dx) / (2.0 * sigma * sigma)) * np.exp(-((np.abs(dy) / length) ** 6))
    elif class_name == "horizontal_bar":
        length = float(rng.uniform(0.8, 1.15))
        mask = np.exp(-(dy * dy) / (2.0 * sigma * sigma)) * np.exp(-((np.abs(dx) / length) ** 6))
    elif class_name == "cross":
        length = float(rng.uniform(0.75, 1.1))
        vertical = np.exp(-(dx * dx) / (2.0 * sigma * sigma)) * np.exp(-((np.abs(dy) / length) ** 6))
        horizontal = np.exp(-(dy * dy) / (2.0 * sigma * sigma)) * np.exp(-((np.abs(dx) / length) ** 6))
        mask = np.maximum(vertical, horizontal)
    elif class_name == "x_shape":
        diag_sigma = sigma * 0.85
        diag1 = np.exp(-(((dy - dx) / np.sqrt(2.0)) ** 2) / (2.0 * diag_sigma * diag_sigma))
        diag2 = np.exp(-(((dy + dx) / np.sqrt(2.0)) ** 2) / (2.0 * diag_sigma * diag_sigma))
        extent = np.exp(-(((np.abs(dx) + np.abs(dy)) / 1.35) ** 6))
        mask = np.maximum(diag1, diag2) * extent
    elif class_name == "diamond":
        radius = float(rng.uniform(0.55, 0.78))
        mask = np.clip(1.0 - (np.abs(dx) + np.abs(dy)) / radius, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown class_name: {class_name}")

    background = 0.08 + 0.05 * (1.0 - 0.5 * (yy + 1.0))
    image = background + float(rng.uniform(0.7, 0.95)) * mask
    image += 0.05 * np.exp(-((dx * dx + dy * dy) / 1.8)).astype(np.float32)
    image += rng.normal(0.0, float(noise_std), size=(size, size)).astype(np.float32)
    return np.clip(image.astype(np.float32), 0.0, 1.0)


class EpisodicFewShotRecognitionDataset:
    def __init__(self, cfg: DataConfig) -> None:
        _validate_config(cfg)
        self.cfg = cfg
        self.episodes = _make_episode_specs(cfg)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int):
        import torch

        episode = self.episodes[int(idx)]
        support_images = np.stack(
            [
                _render_shape(
                    sample.class_name,
                    sample.seed,
                    image_size=int(self.cfg.image_size),
                    noise_std=float(self.cfg.noise_std),
                )
                for sample in episode.support_samples
            ],
            axis=0,
        )
        query_images = np.stack(
            [
                _render_shape(
                    sample.class_name,
                    sample.seed,
                    image_size=int(self.cfg.image_size),
                    noise_std=float(self.cfg.noise_std),
                )
                for sample in episode.query_samples
            ],
            axis=0,
        )
        support_labels = np.asarray([sample.label for sample in episode.support_samples], dtype=np.int64)
        query_labels = np.asarray([sample.label for sample in episode.query_samples], dtype=np.int64)

        return {
            "support_images": torch.from_numpy(support_images).unsqueeze(1),
            "support_labels": torch.from_numpy(support_labels),
            "query_images": torch.from_numpy(query_images).unsqueeze(1),
            "query_labels": torch.from_numpy(query_labels),
            "class_names": list(episode.class_names),
        }


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    dataset = EpisodicFewShotRecognitionDataset(cfg)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    def _collate(batch):
        return {
            "support_images": torch.stack([item["support_images"] for item in batch], dim=0),
            "support_labels": torch.stack([item["support_labels"] for item in batch], dim=0),
            "query_images": torch.stack([item["query_images"] for item in batch], dim=0),
            "query_labels": torch.stack([item["query_labels"] for item in batch], dim=0),
            "class_names": [item["class_names"] for item in batch],
        }

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "EpisodicFewShotRecognitionDataset", "get_dataloaders"]
