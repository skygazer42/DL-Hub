from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

_COLORS: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.18, 0.18),
    "green": (0.2, 0.95, 0.25),
    "blue": (0.2, 0.35, 1.0),
}
_SHAPES: tuple[str, ...] = ("square", "circle", "cross")


@dataclass(frozen=True)
class EventSpec:
    event_name: str
    color_name: str
    shape_name: str
    audio_bin: float


_EVENTS: tuple[EventSpec, ...] = (
    EventSpec("red_square_bell", "red", "square", 0.20),
    EventSpec("green_circle_chime", "green", "circle", 0.55),
    EventSpec("blue_cross_click", "blue", "cross", 0.80),
)


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    @property
    def pad_id(self) -> int:
        return int(self.token_to_id[self.pad_token])

    @property
    def bos_id(self) -> int:
        return int(self.token_to_id[self.bos_token])

    @property
    def eos_id(self) -> int:
        return int(self.token_to_id[self.eos_token])

    @property
    def size(self) -> int:
        return int(len(self.id_to_token))

    def encode_query(self, tokens: list[str], *, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = [self.bos_id, *[int(self.token_to_id[token]) for token in tokens], self.eos_id]
        if len(seq) > int(max_length):
            raise ValueError(f"Query exceeds max_length={int(max_length)}")
        pad_count = int(max_length) - len(seq)
        seq.extend([self.pad_id] * pad_count)
        mask = [1.0] * (int(max_length) - pad_count) + [0.0] * pad_count
        return torch.tensor(seq, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 384
    batch_size: int = 16
    num_frames: int = 6
    image_size: int = 20
    audio_window: int = 12
    max_text_length: int = 12
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "when",
        "does",
        "the",
        "red",
        "green",
        "blue",
        "square",
        "circle",
        "cross",
        "bell",
        "chime",
        "click",
        "happen",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _shape_mask(image_size: int, shape_name: str, center_x: float, center_y: float, radius: float) -> torch.Tensor:
    ys = torch.arange(image_size, dtype=torch.float32).view(-1, 1).expand(image_size, image_size)
    xs = torch.arange(image_size, dtype=torch.float32).view(1, -1).expand(image_size, image_size)

    if shape_name == "square":
        inside_x = (xs - center_x).abs() <= radius
        inside_y = (ys - center_y).abs() <= radius
        return inside_x & inside_y
    if shape_name == "circle":
        return (xs - center_x).pow(2) + (ys - center_y).pow(2) <= radius * radius
    if shape_name == "cross":
        thickness = max(1.0, radius / 2.0)
        vertical = (xs - center_x).abs() <= thickness
        vertical = vertical & ((ys - center_y).abs() <= radius)
        horizontal = (ys - center_y).abs() <= thickness
        horizontal = horizontal & ((xs - center_x).abs() <= radius)
        return vertical | horizontal
    raise ValueError(f"Unsupported shape: {shape_name}")


def _render_video(spec: EventSpec, event_frame: int, cfg: DataConfig) -> torch.Tensor:
    image_size = int(cfg.image_size)
    radius = 2.5
    frames: list[torch.Tensor] = []
    for frame_idx in range(int(cfg.num_frames)):
        image = torch.full((3, image_size, image_size), 0.04, dtype=torch.float32)
        center_x = 0.5 * (image_size - 1) + 2.0 * (float(frame_idx) - float(event_frame))
        center_x = float(max(4.0, min(float(image_size - 5), center_x)))
        center_y = 0.5 * (image_size - 1)
        mask = _shape_mask(image_size, spec.shape_name, center_x, center_y, radius)
        color = torch.tensor(_COLORS[spec.color_name], dtype=torch.float32).view(3, 1, 1)
        intensity = 1.0 if frame_idx == int(event_frame) else 0.55
        image = torch.where(mask.unsqueeze(0), (intensity * color).expand_as(image), image)
        image[:, 0, :] = 0.08
        image[:, -1, :] = 0.08
        image[:, :, 0] = 0.08
        image[:, :, -1] = 0.08
        frames.append(image.clamp(0.0, 1.0))
    return torch.stack(frames, dim=0)


def _render_audio(spec: EventSpec, event_frame: int, cfg: DataConfig) -> torch.Tensor:
    bins = torch.arange(int(cfg.audio_window), dtype=torch.float32)
    peak = float(spec.audio_bin) * float(int(cfg.audio_window) - 1)
    width = 1.2
    base_tone = torch.exp(-0.5 * ((bins - peak) / width).pow(2))
    clip = []
    for frame_idx in range(int(cfg.num_frames)):
        gain = 1.0 if frame_idx == int(event_frame) else 0.35
        clip.append((0.02 + gain * base_tone).clamp(0.0, 1.0))
    return torch.stack(clip, dim=0)


def _query_tokens(spec: EventSpec) -> list[str]:
    sound_word = spec.event_name.split("_")[-1]
    return ["when", "does", "the", spec.color_name, spec.shape_name, sound_word, "happen"]


class ToyAudioVisualEventLocalizationDataset(Dataset):
    def __init__(self, cfg: DataConfig, vocab: Vocab) -> None:
        self.cfg = cfg
        self.vocab = vocab
        self.generator = torch.Generator().manual_seed(int(cfg.seed))

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        event_id = int(torch.randint(len(_EVENTS), (1,), generator=self.generator).item())
        event_frame = int(
            torch.randint(1, int(self.cfg.num_frames) - 1, (1,), generator=self.generator).item()
        )
        spec = _EVENTS[event_id]
        query_tokens = _query_tokens(spec)
        query_ids, attention_mask = self.vocab.encode_query(
            query_tokens, max_length=int(self.cfg.max_text_length)
        )
        event_mask = torch.zeros(int(self.cfg.num_frames), dtype=torch.float32)
        event_mask[event_frame] = 1.0
        return {
            "video": _render_video(spec, event_frame, self.cfg),
            "audio_clip": _render_audio(spec, event_frame, self.cfg),
            "query_ids": query_ids,
            "attention_mask": attention_mask,
            "event_mask": event_mask,
            "event_frame": torch.tensor(event_frame, dtype=torch.long),
            "segment": torch.tensor([event_frame, event_frame], dtype=torch.long),
            "query_text": " ".join(query_tokens),
            "event_name": spec.event_name,
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Vocab]:
    vocab = _build_vocab()
    dataset = ToyAudioVisualEventLocalizationDataset(cfg, vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "video": torch.stack([sample["video"] for sample in batch], dim=0),
            "audio_clip": torch.stack([sample["audio_clip"] for sample in batch], dim=0),
            "query_ids": torch.stack([sample["query_ids"] for sample in batch], dim=0),
            "attention_mask": torch.stack([sample["attention_mask"] for sample in batch], dim=0),
            "event_mask": torch.stack([sample["event_mask"] for sample in batch], dim=0),
            "event_frame": torch.stack([sample["event_frame"] for sample in batch], dim=0),
            "segment": torch.stack([sample["segment"] for sample in batch], dim=0),
            "query_text": [str(sample["query_text"]) for sample in batch],
            "event_name": [str(sample["event_name"]) for sample in batch],
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
