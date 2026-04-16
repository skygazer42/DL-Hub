from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

_COLORS: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.15, 0.15),
    "green": (0.15, 0.95, 0.2),
    "blue": (0.2, 0.35, 1.0),
    "yellow": (1.0, 0.9, 0.15),
}
_SHAPES: tuple[str, ...] = ("square", "circle", "cross")
_MOTIONS: tuple[str, ...] = ("left", "right", "up", "around")
_SIZES: dict[str, float] = {"small": 0.14, "large": 0.22}
_QUERY_TEMPLATES: tuple[str, ...] = (
    "video of {color} {shape} moving {motion}",
    "{color} {shape} moving {motion}",
    "find the {color} {shape} going {motion}",
)


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_token: str = "<pad>"

    @property
    def pad_id(self) -> int:
        return int(self.token_to_id[self.pad_token])

    @property
    def size(self) -> int:
        return int(len(self.id_to_token))

    def encode(self, tokens: list[str], *, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if len(tokens) > int(max_length):
            raise ValueError(
                f"Too many tokens for max_length={int(max_length)}: got {len(tokens)} tokens."
            )

        input_ids = [int(self.token_to_id[token]) for token in tokens]
        pad_count = int(max_length) - len(input_ids)
        input_ids.extend([self.pad_id] * pad_count)
        attention_mask = [1.0] * (int(max_length) - pad_count) + [0.0] * pad_count
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.float32),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "pad_token": str(self.pad_token),
            "pad_id": int(self.pad_id),
            "token_to_id": {k: int(v) for k, v in self.token_to_id.items()},
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 512
    batch_size: int = 32
    num_frames: int = 6
    image_size: int = 20
    max_text_length: int = 12
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "video",
        "of",
        "moving",
        "find",
        "the",
        "going",
        *sorted(_COLORS.keys()),
        *_SHAPES,
        *_MOTIONS,
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _all_concepts() -> list[tuple[str, str, str, str]]:
    return list(product(_COLORS.keys(), _SHAPES, _MOTIONS, _SIZES.keys()))


def _make_concept_list(cfg: DataConfig) -> list[tuple[str, str, str, str]]:
    concepts = _all_concepts()
    if int(cfg.num_samples) <= 0:
        raise ValueError("num_samples must be positive")
    if int(cfg.num_frames) < 4:
        raise ValueError("num_frames must be >= 4")
    if int(cfg.image_size) < 16:
        raise ValueError("image_size must be >= 16")
    if int(cfg.max_text_length) < 5:
        raise ValueError("max_text_length must be >= 5")

    generator = torch.Generator().manual_seed(int(cfg.seed))
    order = torch.randperm(len(concepts), generator=generator).tolist()
    shuffled = [concepts[idx] for idx in order]

    out: list[tuple[str, str, str, str]] = []
    while len(out) < int(cfg.num_samples):
        out.extend(shuffled)
    return out[: int(cfg.num_samples)]


def _trajectory(
    motion_name: str,
    frame_idx: int,
    num_frames: int,
    image_size: int,
) -> tuple[float, float]:
    t = frame_idx / max(1, int(num_frames) - 1)
    low = 0.2 * (int(image_size) - 1)
    high = 0.8 * (int(image_size) - 1)
    mid = 0.5 * (int(image_size) - 1)
    span = high - low

    if motion_name == "left":
        return mid, high - t * span
    if motion_name == "right":
        return mid, low + t * span
    if motion_name == "up":
        return high - t * span, mid
    if motion_name == "around":
        angle = 2.0 * torch.pi * torch.tensor(t, dtype=torch.float32)
        radius = 0.22 * float(int(image_size) - 1)
        return (
            mid + float(radius * torch.sin(angle).item()),
            mid + float(radius * torch.cos(angle).item()),
        )
    raise ValueError(f"Unsupported motion: {motion_name}")


def _render_shape_frame(
    *,
    image_size: int,
    color_name: str,
    shape_name: str,
    size_name: str,
    center_y: float,
    center_x: float,
) -> torch.Tensor:
    image = torch.full((3, int(image_size), int(image_size)), 0.03, dtype=torch.float32)
    color = torch.tensor(_COLORS[color_name], dtype=torch.float32).view(3, 1, 1)
    radius = max(2, int(round(_SIZES[size_name] * int(image_size))))

    ys = torch.arange(int(image_size), dtype=torch.float32).view(-1, 1)
    xs = torch.arange(int(image_size), dtype=torch.float32).view(1, -1)
    yy = ys.expand(int(image_size), int(image_size))
    xx = xs.expand(int(image_size), int(image_size))

    if shape_name == "square":
        mask = (yy - float(center_y)).abs() <= radius
        mask = mask & ((xx - float(center_x)).abs() <= radius)
    elif shape_name == "circle":
        mask = (yy - float(center_y)).pow(2) + (xx - float(center_x)).pow(2) <= float(
            radius * radius
        )
    elif shape_name == "cross":
        thickness = max(1, radius // 2)
        vertical = (xx - float(center_x)).abs() <= thickness
        horizontal = (yy - float(center_y)).abs() <= thickness
        arm = (yy - float(center_y)).abs() <= radius
        arm = arm & ((xx - float(center_x)).abs() <= radius)
        mask = (vertical & arm) | (horizontal & arm)
    else:
        raise ValueError(f"Unsupported shape: {shape_name}")

    image = torch.where(mask.unsqueeze(0), color.expand_as(image), image)
    image[:, 0, :] = 0.08
    image[:, -1, :] = 0.08
    image[:, :, 0] = 0.08
    image[:, :, -1] = 0.08
    return image.clamp(0.0, 1.0)


def _render_video(
    *,
    image_size: int,
    num_frames: int,
    color_name: str,
    shape_name: str,
    motion_name: str,
    size_name: str,
) -> torch.Tensor:
    frames = []
    for frame_idx in range(int(num_frames)):
        center_y, center_x = _trajectory(motion_name, frame_idx, int(num_frames), int(image_size))
        frames.append(
            _render_shape_frame(
                image_size=int(image_size),
                color_name=color_name,
                shape_name=shape_name,
                size_name=size_name,
                center_y=center_y,
                center_x=center_x,
            )
        )
    return torch.stack(frames, dim=0)


class ToyVideoTextRetrievalDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        self.cfg = cfg
        self.vocab = vocab
        self.concepts = _make_concept_list(cfg)

    def __len__(self) -> int:
        return int(len(self.concepts))

    def __getitem__(self, idx: int) -> dict[str, object]:
        color_name, shape_name, motion_name, size_name = self.concepts[int(idx)]
        query_template = _QUERY_TEMPLATES[int(idx) % len(_QUERY_TEMPLATES)]
        query_text = query_template.format(
            color=color_name,
            shape=shape_name,
            motion=motion_name,
        )
        tokens = query_text.split()
        input_ids, attention_mask = self.vocab.encode(tokens, max_length=int(self.cfg.max_text_length))
        video = _render_video(
            image_size=int(self.cfg.image_size),
            num_frames=int(self.cfg.num_frames),
            color_name=color_name,
            shape_name=shape_name,
            motion_name=motion_name,
            size_name=size_name,
        )
        caption_text = f"video of {color_name} {shape_name} moving {motion_name}"
        return {
            "video": video,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pair_id": int(idx),
            "caption_text": caption_text,
            "query_text": query_text,
            "motion_type": motion_name,
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Vocab]:
    vocab = _build_vocab()
    dataset = ToyVideoTextRetrievalDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        video = torch.stack([sample["video"] for sample in batch], dim=0)
        input_ids = torch.stack([sample["input_ids"] for sample in batch], dim=0)
        attention_mask = torch.stack([sample["attention_mask"] for sample in batch], dim=0)
        pair_id = torch.tensor([sample["pair_id"] for sample in batch], dtype=torch.long)
        caption_text = [str(sample["caption_text"]) for sample in batch]
        query_text = [str(sample["query_text"]) for sample in batch]
        motion_type = [str(sample["motion_type"]) for sample in batch]
        return {
            "video": video,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pair_id": pair_id,
            "caption_text": caption_text,
            "query_text": query_text,
            "motion_type": motion_type,
        }

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    return train_loader, val_loader, vocab


__all__ = ["DataConfig", "ToyVideoTextRetrievalDataset", "Vocab", "get_dataloaders"]
