from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

_PROMPT_BANK = (
    "a red toy car turning left",
    "a small robot waving",
    "a paper airplane gliding",
    "a candle flickering in wind",
    "a neon fish swimming",
    "a blue kite in the sky",
    "a toy train crossing",
    "a bouncing tennis ball",
    "a lantern floating upward",
    "a windmill spinning slowly",
)


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 8
    frames: int = 4
    num_workers: int = 0
    num_samples: int = 64
    seed: int = 0
    val_fraction: float = 0.2


def _prompt_to_scalar(prompt: str) -> float:
    encoded = prompt.encode("utf-8")
    if not encoded:
        return 0.0
    return float(sum(encoded) % 251) / 250.0


def _render_video(prompt: str, frames: int) -> torch.Tensor:
    scalar = _prompt_to_scalar(prompt)
    base = torch.tensor(
        [0.15 + 0.75 * scalar, 0.25 + 0.5 * (1.0 - scalar), 0.2 + 0.6 * (0.5 + 0.5 * torch.sin(torch.tensor(scalar)))],
        dtype=torch.float32,
    ).view(1, 3, 1, 1)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 8, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, 8, dtype=torch.float32),
        indexing="ij",
    )
    pattern = torch.stack(
        [
            0.5 + 0.5 * torch.sin((2.0 + scalar) * xx + 0.8 * yy),
            0.5 + 0.5 * torch.cos(1.3 * xx - (1.8 + scalar) * yy),
            0.5 + 0.5 * torch.sin((xx * xx + yy * yy) * (2.0 + scalar)),
        ],
        dim=0,
    )
    seq: list[torch.Tensor] = []
    for frame_idx in range(int(frames)):
        alpha = float(frame_idx) / max(1, int(frames) - 1)
        shift = int(round(alpha * (1.0 + 2.0 * scalar)))
        moved = torch.roll(pattern, shifts=(shift, -shift), dims=(1, 2))
        frame = torch.clamp(base + 0.4 * moved.unsqueeze(0), 0.0, 1.0)
        seq.append(frame * 2.0 - 1.0)
    return torch.cat(seq, dim=0)


class ToyTextToVideoDataset(Dataset):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        generator = torch.Generator().manual_seed(int(cfg.seed))
        prompts: list[str] = []
        videos: list[torch.Tensor] = []
        bank_size = len(_PROMPT_BANK)
        for _ in range(int(cfg.num_samples)):
            idx = int(torch.randint(0, bank_size, (1,), generator=generator).item())
            prompt = _PROMPT_BANK[idx]
            prompts.append(prompt)
            videos.append(_render_video(prompt, int(cfg.frames)))
        self._prompts = prompts
        self._videos = torch.stack(videos, dim=0)

    def __len__(self) -> int:
        return int(self._videos.shape[0])

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor]:
        i = int(idx)
        return self._prompts[i], self._videos[i]


def _collate_batch(batch: list[tuple[str, torch.Tensor]]) -> tuple[list[str], torch.Tensor]:
    prompts = [item[0] for item in batch]
    videos = torch.stack([item[1] for item in batch], dim=0)
    return prompts, videos


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    dataset: Dataset = ToyTextToVideoDataset(cfg)
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
        collate_fn=_collate_batch,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        drop_last=False,
        collate_fn=_collate_batch,
    )
    return train_loader, val_loader


__all__ = ["DataConfig", "ToyTextToVideoDataset", "get_dataloaders"]
