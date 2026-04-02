from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 2
    image_size: int = 64
    seed: int = 0
    in_channels: int = 3

    noise_std: float = 0.15
    square_min: int = 8
    square_max: int = 20
    stripe_period: int = 8


def _make_square_image(rng: np.random.Generator, *, image_size: int, noise_std: float, square_min: int, square_max: int) -> np.ndarray:
    s = int(image_size)
    img = rng.normal(loc=0.0, scale=float(noise_std), size=(s, s)).astype(np.float32)
    img = np.clip(img, -1.0, 1.0)
    size = int(rng.integers(low=int(square_min), high=int(square_max) + 1))
    top = int(rng.integers(low=0, high=max(1, s - size)))
    left = int(rng.integers(low=0, high=max(1, s - size)))
    img[top : top + size, left : left + size] = 1.0
    return img


def _make_stripes_image(rng: np.random.Generator, *, image_size: int, noise_std: float, stripe_period: int) -> np.ndarray:
    s = int(image_size)
    yy, xx = np.meshgrid(np.arange(s, dtype=np.float32), np.arange(s, dtype=np.float32), indexing="ij")
    period = max(2, int(stripe_period))
    stripes = np.sin(2.0 * np.pi * xx / float(period))
    stripes = (stripes > 0).astype(np.float32) * 2.0 - 1.0  # {-1, +1}
    noise = rng.normal(loc=0.0, scale=float(noise_std), size=(s, s)).astype(np.float32)
    img = np.clip(stripes * 0.7 + noise * 0.3, -1.0, 1.0)
    return img


def make_batch(cfg: DataConfig):
    import torch

    rng = np.random.default_rng(int(cfg.seed))
    b = int(cfg.batch_size)
    s = int(cfg.image_size)
    c = int(cfg.in_channels)

    content = np.stack(
        [
            _make_square_image(
                rng,
                image_size=s,
                noise_std=float(cfg.noise_std),
                square_min=int(cfg.square_min),
                square_max=int(cfg.square_max),
            )
            for _ in range(b)
        ],
        axis=0,
    )
    style = np.stack(
        [
            _make_stripes_image(
                rng, image_size=s, noise_std=float(cfg.noise_std), stripe_period=int(cfg.stripe_period)
            )
            for _ in range(b)
        ],
        axis=0,
    )

    content_t = torch.from_numpy(content).unsqueeze(1).repeat(1, c, 1, 1)
    style_t = torch.from_numpy(style).unsqueeze(1).repeat(1, c, 1, 1)
    return content_t.to(torch.float32), style_t.to(torch.float32)


__all__ = ["DataConfig", "make_batch"]

