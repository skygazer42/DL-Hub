from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

_COLORS: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.15, 0.15),
    "green": (0.15, 0.95, 0.2),
    "blue": (0.2, 0.35, 1.0),
    "yellow": (1.0, 0.9, 0.15),
}
_SHAPES: tuple[str, ...] = ("square", "circle", "cross")
_SIZES: tuple[str, ...] = ("small", "large")
_QUADRANTS: tuple[tuple[str, str], ...] = (
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
)
_QUESTION_TYPES: tuple[str, ...] = ("color", "shape", "size", "location", "yes_no")


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    sep_token: str = "<sep>"

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
    def sep_id(self) -> int:
        return int(self.token_to_id[self.sep_token])

    @property
    def size(self) -> int:
        return int(len(self.id_to_token))

    def encode_question(self, tokens: list[str], *, max_length: int) -> torch.Tensor:
        seq = [self.bos_id, *[int(self.token_to_id[token]) for token in tokens], self.sep_id]
        if len(seq) > int(max_length):
            raise ValueError(f"Question exceeds max_length={int(max_length)}.")
        pad_count = int(max_length) - len(seq)
        seq.extend([self.pad_id] * pad_count)
        return torch.tensor(seq, dtype=torch.long)

    def encode_example(
        self, question_tokens: list[str], answer_tokens: list[str], *, max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_ids = [self.bos_id, *[int(self.token_to_id[token]) for token in question_tokens], self.sep_id]
        answer_ids = [int(self.token_to_id[token]) for token in answer_tokens]
        full_ids = [*prompt_ids, *answer_ids]
        if len(full_ids) > int(max_length):
            raise ValueError(f"Example exceeds max_length={int(max_length)}.")

        labels = [-100] * (len(prompt_ids) - 1) + [*answer_ids, self.eos_id]
        pad_count = int(max_length) - len(full_ids)
        full_ids.extend([self.pad_id] * pad_count)
        labels.extend([-100] * pad_count)
        mask = [1.0] * (int(max_length) - pad_count) + [0.0] * pad_count

        return (
            torch.tensor(full_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32),
        )

    def decode_ids(self, ids: list[int]) -> list[str]:
        tokens: list[str] = []
        for idx in ids:
            token = self.id_to_token[int(idx)]
            if token in {self.pad_token, self.bos_token, self.sep_token}:
                continue
            if token == self.eos_token:
                break
            tokens.append(token)
        return tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "pad_id": int(self.pad_id),
            "bos_id": int(self.bos_id),
            "eos_id": int(self.eos_id),
            "sep_id": int(self.sep_id),
            "token_to_id": {k: int(v) for k, v in self.token_to_id.items()},
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 512
    batch_size: int = 32
    image_size: int = 16
    scene_size: int = 32
    max_text_length: int = 14
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "<sep>",
        "what",
        "where",
        "color",
        "shape",
        "size",
        "is",
        "the",
        "object",
        "at",
        "yes",
        "no",
        "top",
        "bottom",
        "left",
        "right",
        "red",
        "green",
        "blue",
        "yellow",
        "square",
        "circle",
        "cross",
        "small",
        "large",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _render_object(
    *,
    canvas: torch.Tensor,
    color_name: str,
    shape_name: str,
    size_name: str,
    location: tuple[str, str],
) -> None:
    color = torch.tensor(_COLORS[color_name], dtype=torch.float32).view(3, 1, 1)
    scene_size = int(canvas.shape[-1])
    vertical, horizontal = location
    center_y = int(round((0.25 if vertical == "top" else 0.75) * (scene_size - 1)))
    center_x = int(round((0.25 if horizontal == "left" else 0.75) * (scene_size - 1)))
    radius = max(3, int(round((0.12 if size_name == "small" else 0.18) * scene_size)))

    ys = torch.arange(scene_size, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(scene_size, dtype=torch.float32).view(1, -1)
    yy = ys.expand(scene_size, scene_size)
    xx = xs.expand(scene_size, scene_size)

    if shape_name == "square":
        mask = (yy - center_y).abs() <= radius
        mask = mask & ((xx - center_x).abs() <= radius)
    elif shape_name == "circle":
        mask = (yy - center_y).pow(2) + (xx - center_x).pow(2) <= float(radius * radius)
    elif shape_name == "cross":
        thickness = max(1, radius // 2)
        vertical_mask = (xx - center_x).abs() <= thickness
        horizontal_mask = (yy - center_y).abs() <= thickness
        arm = (yy - center_y).abs() <= radius
        arm = arm & ((xx - center_x).abs() <= radius)
        mask = (vertical_mask & arm) | (horizontal_mask & arm)
    else:
        raise ValueError(f"Unsupported shape: {shape_name}")

    canvas[:] = torch.where(mask.unsqueeze(0), color.expand_as(canvas), canvas)


def _resize_image(image: torch.Tensor, *, image_size: int) -> torch.Tensor:
    return F.interpolate(
        image.unsqueeze(0),
        size=(int(image_size), int(image_size)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _build_scene(
    *,
    scene_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, list[dict[str, str]]]:
    scene = torch.full((3, int(scene_size), int(scene_size)), 0.03, dtype=torch.float32)
    color_order = torch.randperm(len(_COLORS), generator=generator).tolist()
    object_specs: list[dict[str, str]] = []
    for quad_idx, location in enumerate(_QUADRANTS):
        color_name = tuple(_COLORS.keys())[color_order[quad_idx]]
        shape_name = _SHAPES[int(torch.randint(0, len(_SHAPES), (1,), generator=generator).item())]
        size_name = _SIZES[int(torch.randint(0, len(_SIZES), (1,), generator=generator).item())]
        _render_object(
            canvas=scene,
            color_name=color_name,
            shape_name=shape_name,
            size_name=size_name,
            location=location,
        )
        object_specs.append(
            {
                "vertical": location[0],
                "horizontal": location[1],
                "color": color_name,
                "shape": shape_name,
                "size": size_name,
            }
        )

    scene[:, 0, :] = 0.08
    scene[:, -1, :] = 0.08
    scene[:, :, 0] = 0.08
    scene[:, :, -1] = 0.08
    return scene.clamp(0.0, 1.0), object_specs


def _build_views(scene: torch.Tensor, *, image_size: int) -> torch.Tensor:
    scene_size = int(scene.shape[-1])
    half = scene_size // 2
    quadrants = [
        scene[:, :half, :half],
        scene[:, :half, half:],
        scene[:, half:, :half],
        scene[:, half:, half:],
    ]
    full_view = _resize_image(scene, image_size=int(image_size))
    crop_views = [_resize_image(crop, image_size=int(image_size)) for crop in quadrants]
    return torch.stack([full_view, *crop_views], dim=0)


def _make_question_answer(
    *,
    object_specs: list[dict[str, str]],
    idx: int,
) -> tuple[str, list[str], list[str]]:
    qtype = _QUESTION_TYPES[int(idx) % len(_QUESTION_TYPES)]
    target_idx = (int(idx) // len(_QUESTION_TYPES)) % len(object_specs)
    target = object_specs[target_idx]
    location_tokens = [target["vertical"], target["horizontal"]]

    if qtype == "color":
        return qtype, ["what", "color", "is", "the", "object", "at", *location_tokens], [target["color"]]
    if qtype == "shape":
        return qtype, ["what", "shape", "is", "the", "object", "at", *location_tokens], [target["shape"]]
    if qtype == "size":
        return qtype, ["what", "size", "is", "the", "object", "at", *location_tokens], [target["size"]]
    if qtype == "location":
        return qtype, ["where", "is", "the", target["color"], "object"], location_tokens

    truth = ((int(idx) // (len(_QUESTION_TYPES) * len(object_specs))) % 2) == 0
    color_name = target["color"] if truth else next(name for name in _COLORS if name != target["color"])
    question = ["is", "the", "object", "at", *location_tokens, color_name]
    return "yes_no", question, ["yes" if truth else "no"]


def _build_record(
    *,
    cfg: DataConfig,
    idx: int,
    vocab: Vocab,
    generator: torch.Generator,
) -> dict[str, object]:
    scene, object_specs = _build_scene(scene_size=int(cfg.scene_size), generator=generator)
    question_type, question_tokens, answer_tokens = _make_question_answer(
        object_specs=object_specs,
        idx=int(idx),
    )
    question_ids = vocab.encode_question(question_tokens, max_length=int(cfg.max_text_length))
    input_ids, labels, attention_mask = vocab.encode_example(
        question_tokens,
        answer_tokens,
        max_length=int(cfg.max_text_length),
    )
    views = _build_views(scene, image_size=int(cfg.image_size))
    return {
        "images": views,
        "question_ids": question_ids,
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "question_type": question_type,
        "question_text": " ".join(question_tokens),
        "answer_text": " ".join(answer_tokens),
    }


class ToyPerceiverDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        self.cfg = cfg
        self.vocab = vocab
        if int(cfg.num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        if int(cfg.image_size) < 12:
            raise ValueError("image_size must be >= 12")
        if int(cfg.scene_size) < int(cfg.image_size):
            raise ValueError("scene_size must be >= image_size")
        if int(cfg.max_text_length) < 14:
            raise ValueError("max_text_length must be >= 14")

        generator = torch.Generator().manual_seed(int(cfg.seed))
        self.records = [
            _build_record(cfg=cfg, idx=idx, vocab=vocab, generator=generator)
            for idx in range(int(cfg.num_samples))
        ]

    def __len__(self) -> int:
        return int(len(self.records))

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.records[int(idx)]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Vocab]:
    vocab = _build_vocab()
    dataset = ToyPerceiverDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "images": torch.stack([sample["images"] for sample in batch], dim=0),
            "question_ids": torch.stack([sample["question_ids"] for sample in batch], dim=0),
            "input_ids": torch.stack([sample["input_ids"] for sample in batch], dim=0),
            "labels": torch.stack([sample["labels"] for sample in batch], dim=0),
            "attention_mask": torch.stack([sample["attention_mask"] for sample in batch], dim=0),
            "question_type": [str(sample["question_type"]) for sample in batch],
            "question_text": [str(sample["question_text"]) for sample in batch],
            "answer_text": [str(sample["answer_text"]) for sample in batch],
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


__all__ = ["DataConfig", "ToyPerceiverDataset", "Vocab", "get_dataloaders"]
