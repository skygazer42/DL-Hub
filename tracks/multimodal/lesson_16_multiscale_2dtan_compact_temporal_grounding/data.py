from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

_COLORS: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.2, 0.2),
    "green": (0.2, 0.95, 0.25),
    "blue": (0.2, 0.35, 1.0),
    "yellow": (1.0, 0.9, 0.2),
}
_SHAPES: tuple[str, ...] = ("square", "circle", "cross")
_EVENT_TYPES: tuple[str, ...] = ("move_left", "move_right", "flash")


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
            raise ValueError(f"Query exceeds max_length={int(max_length)}.")
        pad_count = int(max_length) - len(seq)
        mask = [1.0] * len(seq) + [0.0] * pad_count
        seq.extend([self.pad_id] * pad_count)
        return torch.tensor(seq, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "pad_id": int(self.pad_id),
            "bos_id": int(self.bos_id),
            "eos_id": int(self.eos_id),
            "token_to_id": {k: int(v) for k, v in self.token_to_id.items()},
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 512
    batch_size: int = 32
    num_frames: int = 8
    image_size: int = 20
    max_text_length: int = 16
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
        "yellow",
        "square",
        "circle",
        "cross",
        "move",
        "left",
        "right",
        "flash",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _shape_mask(
    *,
    height: int,
    width: int,
    shape_name: str,
    center_x: float,
    center_y: float,
    radius: float,
) -> torch.Tensor:
    ys = torch.arange(height, dtype=torch.float32).view(-1, 1).expand(height, width)
    xs = torch.arange(width, dtype=torch.float32).view(1, -1).expand(height, width)

    if shape_name == "square":
        mask = (xs - float(center_x)).abs() <= float(radius)
        return mask & ((ys - float(center_y)).abs() <= float(radius))

    if shape_name == "circle":
        return (xs - float(center_x)).pow(2) + (ys - float(center_y)).pow(2) <= float(radius * radius)

    if shape_name == "cross":
        thickness = max(1.0, float(radius) / 2.0)
        vertical = (xs - float(center_x)).abs() <= thickness
        vertical = vertical & ((ys - float(center_y)).abs() <= float(radius))
        horizontal = (ys - float(center_y)).abs() <= thickness
        horizontal = horizontal & ((xs - float(center_x)).abs() <= float(radius))
        return vertical | horizontal

    raise ValueError(f"Unsupported shape: {shape_name}")


def _render_frame(
    *,
    image_size: int,
    color_name: str,
    shape_name: str,
    center_x: float,
    center_y: float,
    radius: float,
    flash_on: bool,
    generator: torch.Generator,
) -> torch.Tensor:
    image = 0.02 * torch.rand((3, int(image_size), int(image_size)), generator=generator, dtype=torch.float32)
    image = image + 0.03
    mask = _shape_mask(
        height=int(image_size),
        width=int(image_size),
        shape_name=shape_name,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
    )
    color = torch.tensor(_COLORS[color_name], dtype=torch.float32).view(3, 1, 1)
    if flash_on:
        color = (0.65 * color + 0.35 * torch.ones_like(color)).clamp(0.0, 1.0)
    image[:] = torch.where(mask.unsqueeze(0), color.expand_as(image), image)
    image[:, 0, :] = 0.08
    image[:, -1, :] = 0.08
    image[:, :, 0] = 0.08
    image[:, :, -1] = 0.08
    return image.clamp(0.0, 1.0)


def _temporal_iou(start_idx: int, end_idx: int, gt_start: int, gt_end: int) -> float:
    inter = max(0, min(int(end_idx), int(gt_end)) - max(int(start_idx), int(gt_start)) + 1)
    if inter == 0:
        return 0.0
    union = (int(end_idx) - int(start_idx) + 1) + (int(gt_end) - int(gt_start) + 1) - inter
    return float(inter) / float(union)


def _build_scale_map_targets(
    *,
    num_frames: int,
    gt_start: int,
    gt_end: int,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_len = int(num_frames) // int(stride)
    map_labels = torch.zeros((scale_len, scale_len), dtype=torch.float32)
    map_mask = torch.zeros((scale_len, scale_len), dtype=torch.float32)
    for start_idx in range(scale_len):
        for end_idx in range(start_idx, scale_len):
            seg_start = int(start_idx) * int(stride)
            seg_end = min(int(num_frames) - 1, (int(end_idx) + 1) * int(stride) - 1)
            map_mask[start_idx, end_idx] = 1.0
            map_labels[start_idx, end_idx] = _temporal_iou(seg_start, seg_end, gt_start, gt_end)
    return map_labels, map_mask


def _build_multiscale_targets(num_frames: int, gt_start: int, gt_end: int) -> dict[str, torch.Tensor]:
    map_labels_s1, map_mask_s1 = _build_scale_map_targets(
        num_frames=num_frames,
        gt_start=gt_start,
        gt_end=gt_end,
        stride=1,
    )
    map_labels_s2, map_mask_s2 = _build_scale_map_targets(
        num_frames=num_frames,
        gt_start=gt_start,
        gt_end=gt_end,
        stride=2,
    )
    map_labels_s3, map_mask_s3 = _build_scale_map_targets(
        num_frames=num_frames,
        gt_start=gt_start,
        gt_end=gt_end,
        stride=4,
    )
    return {
        "map_labels_s1": map_labels_s1,
        "map_mask_s1": map_mask_s1,
        "map_labels_s2": map_labels_s2,
        "map_mask_s2": map_mask_s2,
        "map_labels_s3": map_labels_s3,
        "map_mask_s3": map_mask_s3,
    }


def _sample_segment(cfg: DataConfig, generator: torch.Generator) -> tuple[int, int]:
    num_frames = int(cfg.num_frames)
    max_start = num_frames - 3
    gt_start = int(torch.randint(1, max_start + 1, (1,), generator=generator).item())
    max_len = min(4, num_frames - gt_start)
    seg_len = int(torch.randint(2, max_len + 1, (1,), generator=generator).item())
    gt_end = int(gt_start + seg_len - 1)
    return gt_start, gt_end


def _sample_positions(
    *,
    cfg: DataConfig,
    event_type: str,
    gt_start: int,
    gt_end: int,
    generator: torch.Generator,
) -> tuple[list[float], list[float]]:
    num_frames = int(cfg.num_frames)
    image_size = float(cfg.image_size)
    radius = 2.5
    margin = radius + 2.0
    travel = max(2.0, 1.5 * float(gt_end - gt_start + 1))
    center_y = float(torch.empty(1).uniform_(margin, image_size - margin, generator=generator).item())

    if event_type == "move_left":
        start_x = float(
            torch.empty(1).uniform_(margin + travel, image_size - margin, generator=generator).item()
        )
        xs: list[float] = []
        for frame_idx in range(num_frames):
            if frame_idx < gt_start:
                xs.append(start_x)
            elif frame_idx <= gt_end:
                progress = float(frame_idx - gt_start + 1) / float(gt_end - gt_start + 1)
                xs.append(start_x - travel * progress)
            else:
                xs.append(start_x - travel)
        ys = [center_y for _ in range(num_frames)]
        return xs, ys

    if event_type == "move_right":
        start_x = float(
            torch.empty(1).uniform_(margin, image_size - margin - travel, generator=generator).item()
        )
        xs: list[float] = []
        for frame_idx in range(num_frames):
            if frame_idx < gt_start:
                xs.append(start_x)
            elif frame_idx <= gt_end:
                progress = float(frame_idx - gt_start + 1) / float(gt_end - gt_start + 1)
                xs.append(start_x + travel * progress)
            else:
                xs.append(start_x + travel)
        ys = [center_y for _ in range(num_frames)]
        return xs, ys

    if event_type == "flash":
        center_x = float(torch.empty(1).uniform_(margin, image_size - margin, generator=generator).item())
        xs = [center_x for _ in range(num_frames)]
        ys = [center_y for _ in range(num_frames)]
        return xs, ys

    raise ValueError(f"Unsupported event_type: {event_type}")


def _make_query_tokens(color_name: str, shape_name: str, event_type: str) -> list[str]:
    if event_type == "move_left":
        return ["when", "does", "the", color_name, shape_name, "move", "left"]
    if event_type == "move_right":
        return ["when", "does", "the", color_name, shape_name, "move", "right"]
    if event_type == "flash":
        return ["when", "does", "the", color_name, shape_name, "flash"]
    raise ValueError(f"Unsupported event_type: {event_type}")


def _build_record(
    *,
    cfg: DataConfig,
    vocab: Vocab,
    idx: int,
    generator: torch.Generator,
) -> dict[str, object]:
    color_name = tuple(_COLORS.keys())[int(torch.randint(0, len(_COLORS), (1,), generator=generator).item())]
    shape_name = _SHAPES[int(torch.randint(0, len(_SHAPES), (1,), generator=generator).item())]
    event_type = _EVENT_TYPES[int(idx) % len(_EVENT_TYPES)]
    gt_start, gt_end = _sample_segment(cfg, generator)
    query_tokens = _make_query_tokens(color_name, shape_name, event_type)
    query_ids, attention_mask = vocab.encode_query(query_tokens, max_length=int(cfg.max_text_length))
    multiscale_targets = _build_multiscale_targets(int(cfg.num_frames), gt_start, gt_end)
    xs, ys = _sample_positions(
        cfg=cfg,
        event_type=event_type,
        gt_start=gt_start,
        gt_end=gt_end,
        generator=generator,
    )

    frames = []
    for frame_idx in range(int(cfg.num_frames)):
        flash_on = event_type == "flash" and gt_start <= frame_idx <= gt_end
        frames.append(
            _render_frame(
                image_size=int(cfg.image_size),
                color_name=color_name,
                shape_name=shape_name,
                center_x=xs[frame_idx],
                center_y=ys[frame_idx],
                radius=2.5,
                flash_on=flash_on,
                generator=generator,
            )
        )

    return {
        "video": torch.stack(frames, dim=0),
        "query_ids": query_ids,
        "attention_mask": attention_mask,
        **multiscale_targets,
        "segment": torch.tensor([gt_start, gt_end], dtype=torch.long),
        "query_text": " ".join(query_tokens),
        "event_type": event_type,
    }


class SyntheticMultiScaleTwoDtanTemporalGroundingDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        self.cfg = cfg
        self.vocab = vocab
        if int(cfg.num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        if int(cfg.num_frames) < 8:
            raise ValueError("num_frames must be >= 8")
        if int(cfg.num_frames) % 4 != 0:
            raise ValueError("num_frames must be divisible by 4")
        if int(cfg.image_size) < 20:
            raise ValueError("image_size must be >= 20")
        if int(cfg.max_text_length) < 12:
            raise ValueError("max_text_length must be >= 12")

        generator = torch.Generator().manual_seed(int(cfg.seed))
        self.records = [
            _build_record(cfg=cfg, vocab=vocab, idx=idx, generator=generator)
            for idx in range(int(cfg.num_samples))
        ]

    def __len__(self) -> int:
        return int(len(self.records))

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.records[int(idx)]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Vocab]:
    vocab = _build_vocab()
    dataset = SyntheticMultiScaleTwoDtanTemporalGroundingDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "video": torch.stack([sample["video"] for sample in batch], dim=0),
            "query_ids": torch.stack([sample["query_ids"] for sample in batch], dim=0),
            "attention_mask": torch.stack([sample["attention_mask"] for sample in batch], dim=0),
            "map_labels_s1": torch.stack([sample["map_labels_s1"] for sample in batch], dim=0),
            "map_mask_s1": torch.stack([sample["map_mask_s1"] for sample in batch], dim=0),
            "map_labels_s2": torch.stack([sample["map_labels_s2"] for sample in batch], dim=0),
            "map_mask_s2": torch.stack([sample["map_mask_s2"] for sample in batch], dim=0),
            "map_labels_s3": torch.stack([sample["map_labels_s3"] for sample in batch], dim=0),
            "map_mask_s3": torch.stack([sample["map_mask_s3"] for sample in batch], dim=0),
            "segment": torch.stack([sample["segment"] for sample in batch], dim=0),
            "query_text": [str(sample["query_text"]) for sample in batch],
            "event_type": [str(sample["event_type"]) for sample in batch],
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


__all__ = ["DataConfig", "SyntheticMultiScaleTwoDtanTemporalGroundingDataset", "Vocab", "get_dataloaders"]
