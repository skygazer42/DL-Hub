from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

_COLORS: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.15, 0.15),
    "green": (0.15, 0.95, 0.25),
    "blue": (0.2, 0.35, 1.0),
    "yellow": (1.0, 0.9, 0.2),
}
_SHAPES: tuple[str, ...] = ("square", "circle", "cross")
_MOTIONS: tuple[str, ...] = ("left", "right", "up", "around")
_AUDIO_PATTERNS: tuple[str, ...] = ("rising", "falling", "staccato", "ripple")
_SEGMENTS: tuple[str, ...] = ("intro", "middle", "outro")


@dataclass(frozen=True)
class EventSpec:
    event_name: str
    color_name: str
    shape_name: str
    motion_name: str
    audio_pattern: str
    size_scale: float
    base_bin: float


_EVENTS: tuple[EventSpec, ...] = (
    EventSpec("red_square_siren", "red", "square", "left", "rising", 0.18, 0.18),
    EventSpec("green_circle_hum", "green", "circle", "right", "falling", 0.19, 0.72),
    EventSpec("blue_cross_click", "blue", "cross", "up", "staccato", 0.18, 0.44),
    EventSpec("yellow_square_spin", "yellow", "square", "around", "ripple", 0.22, 0.56),
    EventSpec("blue_circle_beacon", "blue", "circle", "left", "rising", 0.16, 0.30),
    EventSpec("red_cross_alarm", "red", "cross", "right", "falling", 0.21, 0.66),
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
    num_samples: int = 384
    batch_size: int = 16
    num_frames: int = 6
    image_size: int = 20
    num_mel_bins: int = 24
    num_audio_steps: int = 12
    max_text_length: int = 12
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "find",
        "the",
        "segment",
        "where",
        "audio",
        "video",
        "in",
        "shows",
        *_SEGMENTS,
        *sorted(_COLORS.keys()),
        *_SHAPES,
        *_MOTIONS,
        *_AUDIO_PATTERNS,
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _validate_config(cfg: DataConfig) -> None:
    if int(cfg.num_samples) <= 0:
        raise ValueError("num_samples must be positive")
    if int(cfg.num_frames) < 4:
        raise ValueError("num_frames must be >= 4")
    if int(cfg.image_size) < 16:
        raise ValueError("image_size must be >= 16")
    if int(cfg.num_mel_bins) < 16:
        raise ValueError("num_mel_bins must be >= 16")
    if int(cfg.num_audio_steps) < 8:
        raise ValueError("num_audio_steps must be >= 8")
    if int(cfg.max_text_length) < 10:
        raise ValueError("max_text_length must be >= 10")


def _build_samples(cfg: DataConfig) -> list[tuple[int, int, int]]:
    _validate_config(cfg)
    base = [
        (event_id, segment_id, variation_id)
        for variation_id in range(4)
        for segment_id in range(len(_SEGMENTS))
        for event_id in range(len(_EVENTS))
    ]
    generator = torch.Generator().manual_seed(int(cfg.seed))
    order = torch.randperm(len(base), generator=generator).tolist()
    shuffled = [base[idx] for idx in order]
    samples: list[tuple[int, int, int]] = []
    while len(samples) < int(cfg.num_samples):
        samples.extend(shuffled)
    return samples[: int(cfg.num_samples)]


def _trajectory(motion_name: str, frame_idx: int, num_frames: int, image_size: int) -> tuple[float, float]:
    t = float(frame_idx) / max(1.0, float(num_frames - 1))
    low = 0.22 * (int(image_size) - 1)
    high = 0.78 * (int(image_size) - 1)
    mid = 0.5 * (int(image_size) - 1)
    span = high - low

    if motion_name == "left":
        return mid, high - t * span
    if motion_name == "right":
        return mid, low + t * span
    if motion_name == "up":
        return high - t * span, mid

    angle = 2.0 * torch.pi * torch.tensor(t, dtype=torch.float32)
    radius = 0.2 * float(int(image_size) - 1)
    return (
        mid + float(radius * torch.sin(angle).item()),
        mid + float(radius * torch.cos(angle).item()),
    )


def _render_frame(
    *,
    image_size: int,
    color_name: str,
    shape_name: str,
    size_scale: float,
    center_y: float,
    center_x: float,
) -> torch.Tensor:
    image = torch.full((3, int(image_size), int(image_size)), 0.04, dtype=torch.float32)
    color = torch.tensor(_COLORS[color_name], dtype=torch.float32).view(3, 1, 1)
    radius = max(2, int(round(float(size_scale) * int(image_size))))

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


def _segment_shift(segment_id: int) -> int:
    shifts = (0, 1, 2)
    return int(shifts[int(segment_id)])


def _render_video(spec: EventSpec, cfg: DataConfig, segment_id: int, variation_id: int) -> torch.Tensor:
    frames = []
    offset = (_segment_shift(segment_id) + int(variation_id)) % int(cfg.num_frames)
    size_scale = float(spec.size_scale) + 0.01 * float(int(segment_id) - 1)
    for frame_idx in range(int(cfg.num_frames)):
        shifted_idx = (int(frame_idx) + offset) % int(cfg.num_frames)
        center_y, center_x = _trajectory(
            spec.motion_name,
            shifted_idx,
            int(cfg.num_frames),
            int(cfg.image_size),
        )
        frames.append(
            _render_frame(
                image_size=int(cfg.image_size),
                color_name=spec.color_name,
                shape_name=spec.shape_name,
                size_scale=size_scale,
                center_y=center_y,
                center_x=center_x,
            )
        )
    return torch.stack(frames, dim=0)


def _render_audio(spec: EventSpec, cfg: DataConfig, segment_id: int, variation_id: int) -> torch.Tensor:
    mel = torch.arange(int(cfg.num_mel_bins), dtype=torch.float32).view(-1, 1)
    time = torch.linspace(0.0, 1.0, steps=int(cfg.num_audio_steps), dtype=torch.float32).view(1, -1)
    base = float(spec.base_bin) * float(int(cfg.num_mel_bins) - 1)
    swing = 0.18 * float(int(cfg.num_mel_bins) - 1)
    phase = 0.12 * float(int(variation_id))
    segment_gate = (0.85, 1.0, 0.75)[int(segment_id)]

    if spec.audio_pattern == "rising":
        center = base - 0.5 * swing + swing * (time + phase).clamp(0.0, 1.0)
        gate = torch.ones_like(time)
    elif spec.audio_pattern == "falling":
        center = base + 0.5 * swing - swing * (time + phase).clamp(0.0, 1.0)
        gate = torch.ones_like(time)
    elif spec.audio_pattern == "staccato":
        center = base + 0.2 * swing * torch.sin(2.0 * torch.pi * (time + phase))
        steps = torch.arange(int(cfg.num_audio_steps), dtype=torch.float32).view(1, -1)
        gate = ((steps + int(segment_id) + int(variation_id)) % 3 != 0).to(torch.float32)
    elif spec.audio_pattern == "ripple":
        center = base + 0.35 * swing * torch.sin(2.0 * torch.pi * (time + phase))
        gate = 0.65 + 0.35 * torch.cos(4.0 * torch.pi * (time + phase))
    else:
        raise ValueError(f"Unsupported audio pattern: {spec.audio_pattern}")

    width = 1.3 + 0.1 * float(variation_id)
    energy = torch.exp(-0.5 * ((mel - center) / width).pow(2))
    spectrogram = 0.02 + 0.9 * energy * gate * float(segment_gate)
    return spectrogram.unsqueeze(0).clamp(0.0, 1.0)


class ToyAudioGroundedRetrievalDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        self.cfg = cfg
        self.vocab = vocab
        self.samples = _build_samples(cfg)

    def __len__(self) -> int:
        return int(len(self.samples))

    def __getitem__(self, idx: int) -> dict[str, object]:
        event_id, segment_id, variation_id = self.samples[int(idx)]
        spec = _EVENTS[int(event_id)]
        segment_name = _SEGMENTS[int(segment_id)]

        query_tokens = [
            "find",
            "the",
            "segment",
            segment_name,
            "where",
            "audio",
            spec.audio_pattern,
            "video",
            "shows",
            spec.motion_name,
            spec.shape_name,
            spec.color_name,
        ]
        input_ids, attention_mask = self.vocab.encode(
            query_tokens,
            max_length=int(self.cfg.max_text_length),
        )
        query_text = " ".join(query_tokens)
        motion_id = _MOTIONS.index(spec.motion_name)

        return {
            "video": _render_video(spec, self.cfg, segment_id, variation_id),
            "audio_spectrogram": _render_audio(spec, self.cfg, segment_id, variation_id),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pair_id": torch.tensor(int(idx), dtype=torch.long),
            "segment_id": torch.tensor(int(segment_id), dtype=torch.long),
            "motion_id": torch.tensor(int(motion_id), dtype=torch.long),
            "event_name": spec.event_name,
            "motion_name": spec.motion_name,
            "query_text": query_text,
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Vocab]:
    vocab = _build_vocab()
    dataset = ToyAudioGroundedRetrievalDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "video": torch.stack([sample["video"] for sample in batch], dim=0),
            "audio_spectrogram": torch.stack([sample["audio_spectrogram"] for sample in batch], dim=0),
            "input_ids": torch.stack([sample["input_ids"] for sample in batch], dim=0),
            "attention_mask": torch.stack([sample["attention_mask"] for sample in batch], dim=0),
            "pair_id": torch.stack([sample["pair_id"] for sample in batch], dim=0),
            "segment_id": torch.stack([sample["segment_id"] for sample in batch], dim=0),
            "motion_id": torch.stack([sample["motion_id"] for sample in batch], dim=0),
            "event_name": [str(sample["event_name"]) for sample in batch],
            "motion_name": [str(sample["motion_name"]) for sample in batch],
            "query_text": [str(sample["query_text"]) for sample in batch],
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


__all__ = [
    "DataConfig",
    "EventSpec",
    "ToyAudioGroundedRetrievalDataset",
    "Vocab",
    "get_dataloaders",
]
