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
_SIZES: tuple[str, ...] = ("small", "large")
_LOCATIONS: tuple[tuple[str, str], ...] = (
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
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

    def encode_caption(
        self, tokens: list[str], *, max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_ids = [int(self.token_to_id[token]) for token in tokens]
        caption_in = [self.bos_id, *token_ids]
        caption_out = [*token_ids, self.eos_id]
        if len(caption_in) > int(max_length) or len(caption_out) > int(max_length):
            raise ValueError(
                f"Caption exceeds max_length={int(max_length)} with {len(tokens)} tokens."
            )

        pad_count = int(max_length) - len(caption_in)
        caption_in.extend([self.pad_id] * pad_count)
        caption_out.extend([self.pad_id] * pad_count)
        mask = [1.0] * (int(max_length) - pad_count) + [0.0] * pad_count
        return (
            torch.tensor(caption_in, dtype=torch.long),
            torch.tensor(caption_out, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32),
        )

    def encode_sequence(
        self, tokens: list[str], *, max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq = [self.bos_id, *[int(self.token_to_id[token]) for token in tokens], self.eos_id]
        if len(seq) > int(max_length):
            raise ValueError(f"Sequence exceeds max_length={int(max_length)}.")

        pad_count = int(max_length) - len(seq)
        seq.extend([self.pad_id] * pad_count)
        mask = [1.0] * (int(max_length) - pad_count) + [0.0] * pad_count
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32),
        )

    def decode_ids(self, ids: list[int]) -> list[str]:
        tokens: list[str] = []
        for idx in ids:
            token = self.id_to_token[int(idx)]
            if token in {self.pad_token, self.bos_token}:
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
            "token_to_id": {k: int(v) for k, v in self.token_to_id.items()},
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 512
    batch_size: int = 32
    image_size: int = 16
    max_text_length: int = 10
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    negative_fraction: float = 0.5


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "a",
        "small",
        "large",
        "red",
        "green",
        "blue",
        "yellow",
        "square",
        "circle",
        "cross",
        "at",
        "top",
        "bottom",
        "left",
        "right",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _all_concepts() -> list[tuple[str, str, str, tuple[str, str]]]:
    return list(product(_COLORS.keys(), _SHAPES, _SIZES, _LOCATIONS))


def _make_concepts(cfg: DataConfig) -> list[tuple[str, str, str, tuple[str, str]]]:
    concepts = _all_concepts()
    if int(cfg.num_samples) <= 0:
        raise ValueError("num_samples must be positive")
    if int(cfg.image_size) < 12:
        raise ValueError("image_size must be >= 12")
    if int(cfg.max_text_length) < 9:
        raise ValueError("max_text_length must be >= 9")
    if not 0.0 <= float(cfg.negative_fraction) <= 1.0:
        raise ValueError("negative_fraction must be in [0, 1]")

    generator = torch.Generator().manual_seed(int(cfg.seed))
    order = torch.randperm(len(concepts), generator=generator).tolist()
    shuffled = [concepts[idx] for idx in order]
    out: list[tuple[str, str, str, tuple[str, str]]] = []
    while len(out) < int(cfg.num_samples):
        out.extend(shuffled)
    return out[: int(cfg.num_samples)]


def _caption_tokens(concept: tuple[str, str, str, tuple[str, str]]) -> list[str]:
    color_name, shape_name, size_name, (vertical, horizontal) = concept
    return ["a", size_name, color_name, shape_name, "at", vertical, horizontal]


def _mutate_concept(
    concept: tuple[str, str, str, tuple[str, str]], *, generator: torch.Generator
) -> tuple[str, str, str, tuple[str, str]]:
    color_name, shape_name, size_name, location = concept
    pick = int(torch.randint(0, 4, (1,), generator=generator).item())

    if pick == 0:
        choices = [name for name in _COLORS if name != color_name]
        color_name = choices[int(torch.randint(0, len(choices), (1,), generator=generator).item())]
    elif pick == 1:
        choices = [name for name in _SHAPES if name != shape_name]
        shape_name = choices[int(torch.randint(0, len(choices), (1,), generator=generator).item())]
    elif pick == 2:
        choices = [name for name in _SIZES if name != size_name]
        size_name = choices[int(torch.randint(0, len(choices), (1,), generator=generator).item())]
    else:
        choices = [name for name in _LOCATIONS if name != location]
        location = choices[int(torch.randint(0, len(choices), (1,), generator=generator).item())]

    return color_name, shape_name, size_name, location


def _render_shape(
    *,
    image_size: int,
    color_name: str,
    shape_name: str,
    size_name: str,
    location: tuple[str, str],
) -> torch.Tensor:
    image = torch.zeros(3, int(image_size), int(image_size), dtype=torch.float32)
    color = torch.tensor(_COLORS[color_name], dtype=torch.float32).view(3, 1, 1)

    vertical, horizontal = location
    center_y = int(round((0.28 if vertical == "top" else 0.72) * (int(image_size) - 1)))
    center_x = int(round((0.28 if horizontal == "left" else 0.72) * (int(image_size) - 1)))
    radius = max(2, int(round((0.16 if size_name == "small" else 0.26) * int(image_size))))

    ys = torch.arange(int(image_size), dtype=torch.float32).view(-1, 1)
    xs = torch.arange(int(image_size), dtype=torch.float32).view(1, -1)
    yy = ys.expand(int(image_size), int(image_size))
    xx = xs.expand(int(image_size), int(image_size))

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

    image[:] = 0.03
    image = torch.where(mask.unsqueeze(0), color.expand_as(image), image)
    image[:, 0, :] = 0.08
    image[:, -1, :] = 0.08
    image[:, :, 0] = 0.08
    image[:, :, -1] = 0.08
    return image.clamp(0.0, 1.0)


class SyntheticBLIPDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        self.cfg = cfg
        self.vocab = vocab
        self.concepts = _make_concepts(cfg)
        self.generator = torch.Generator().manual_seed(int(cfg.seed) + 17)

    def __len__(self) -> int:
        return int(len(self.concepts))

    def __getitem__(self, idx: int) -> dict[str, object]:
        concept = self.concepts[int(idx)]
        color_name, shape_name, size_name, location = concept
        caption_tokens = _caption_tokens(concept)
        caption_in_ids, caption_out_ids, caption_mask = self.vocab.encode_caption(
            caption_tokens, max_length=int(self.cfg.max_text_length)
        )

        itm_tokens = caption_tokens
        label = 1
        use_negative = float(torch.rand(1, generator=self.generator).item()) < float(
            self.cfg.negative_fraction
        )
        if use_negative:
            itm_tokens = _caption_tokens(_mutate_concept(concept, generator=self.generator))
            label = 0

        itm_input_ids, itm_attention_mask = self.vocab.encode_sequence(
            itm_tokens, max_length=int(self.cfg.max_text_length)
        )

        image = _render_shape(
            image_size=int(self.cfg.image_size),
            color_name=color_name,
            shape_name=shape_name,
            size_name=size_name,
            location=location,
        )
        return {
            "image": image,
            "caption_in_ids": caption_in_ids,
            "caption_out_ids": caption_out_ids,
            "caption_mask": caption_mask,
            "itm_input_ids": itm_input_ids,
            "itm_attention_mask": itm_attention_mask,
            "itm_label": int(label),
            "caption_text": " ".join(caption_tokens),
            "itm_text": " ".join(itm_tokens),
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Vocab]:
    vocab = _build_vocab()
    dataset = SyntheticBLIPDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "image": torch.stack([sample["image"] for sample in batch], dim=0),
            "caption_in_ids": torch.stack([sample["caption_in_ids"] for sample in batch], dim=0),
            "caption_out_ids": torch.stack([sample["caption_out_ids"] for sample in batch], dim=0),
            "caption_mask": torch.stack([sample["caption_mask"] for sample in batch], dim=0),
            "itm_input_ids": torch.stack([sample["itm_input_ids"] for sample in batch], dim=0),
            "itm_attention_mask": torch.stack(
                [sample["itm_attention_mask"] for sample in batch], dim=0
            ),
            "itm_label": torch.tensor([sample["itm_label"] for sample in batch], dtype=torch.long),
            "caption_text": [str(sample["caption_text"]) for sample in batch],
            "itm_text": [str(sample["itm_text"]) for sample in batch],
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


__all__ = ["DataConfig", "SyntheticBLIPDataset", "Vocab", "get_dataloaders"]
