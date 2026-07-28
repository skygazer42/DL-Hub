from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

_COLORS: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.2, 0.2),
    "green": (0.2, 0.9, 0.25),
    "blue": (0.2, 0.35, 1.0),
}
_SHAPES: tuple[str, ...] = ("circle", "square", "triangle")
_TEXTURES: tuple[str, ...] = ("solid", "striped")


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
    batch_size: int = 16
    image_size: int = 20
    num_candidates: int = 5
    max_text_length: int = 10
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "photo",
        "of",
        "a",
        *_COLORS.keys(),
        *_TEXTURES,
        *_SHAPES,
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _all_concepts() -> list[tuple[str, str, str]]:
    return [(color, texture, shape) for color, shape, texture in product(_COLORS.keys(), _SHAPES, _TEXTURES)]


def _validate_config(cfg: DataConfig) -> None:
    if int(cfg.num_samples) <= 0:
        raise ValueError("num_samples must be positive")
    if int(cfg.image_size) < 16:
        raise ValueError("image_size must be >= 16")
    if int(cfg.num_candidates) < 2:
        raise ValueError("num_candidates must be >= 2")
    if int(cfg.max_text_length) < 4:
        raise ValueError("max_text_length must be >= 4")


def _make_concept_list(cfg: DataConfig) -> list[tuple[str, str, str]]:
    concepts = _all_concepts()
    gen = torch.Generator().manual_seed(int(cfg.seed))
    order = torch.randperm(len(concepts), generator=gen).tolist()
    shuffled = [concepts[idx] for idx in order]
    out: list[tuple[str, str, str]] = []
    while len(out) < int(cfg.num_samples):
        out.extend(shuffled)
    return out[: int(cfg.num_samples)]


def _render_image(color_name: str, texture_name: str, shape_name: str, image_size: int) -> torch.Tensor:
    size = int(image_size)
    image = torch.full((3, size, size), 0.04, dtype=torch.float32)
    color = torch.tensor(_COLORS[color_name], dtype=torch.float32).view(3, 1, 1)

    ys = torch.arange(size, dtype=torch.float32).view(-1, 1).expand(size, size)
    xs = torch.arange(size, dtype=torch.float32).view(1, -1).expand(size, size)
    cy, cx = 0.5 * (size - 1), 0.5 * (size - 1)
    radius = max(3, size // 4)

    if shape_name == "circle":
        base_mask = (ys - cy).pow(2) + (xs - cx).pow(2) <= float(radius * radius)
    elif shape_name == "square":
        base_mask = (ys - cy).abs() <= radius
        base_mask = base_mask & ((xs - cx).abs() <= radius)
    elif shape_name == "triangle":
        y0 = cy - radius
        y1 = cy + radius
        norm_y = ((ys - y0) / max(1.0, y1 - y0)).clamp(0.0, 1.0)
        half_width = norm_y * radius
        base_mask = (ys >= y0) & (ys <= y1) & ((xs - cx).abs() <= half_width)
    else:
        raise ValueError(f"Unsupported shape: {shape_name}")

    if texture_name == "striped":
        stripe_mask = ((xs.to(torch.long) + ys.to(torch.long)) % 4) < 2
        mask = base_mask & stripe_mask
    elif texture_name == "solid":
        mask = base_mask
    else:
        raise ValueError(f"Unsupported texture: {texture_name}")

    image = torch.where(mask.unsqueeze(0), color.expand_as(image), image)
    image[:, 0, :] = 0.08
    image[:, -1, :] = 0.08
    image[:, :, 0] = 0.08
    image[:, :, -1] = 0.08
    return image.clamp(0.0, 1.0)


def _caption_tokens(concept: tuple[str, str, str]) -> list[str]:
    color_name, texture_name, shape_name = concept
    return ["photo", "of", "a", color_name, texture_name, shape_name]


class SyntheticImageTextRerankingDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        _validate_config(cfg)
        self.cfg = cfg
        self.vocab = vocab
        self.concepts = _make_concept_list(cfg)
        self.all_concepts = _all_concepts()

    def __len__(self) -> int:
        return int(len(self.concepts))

    def __getitem__(self, idx: int) -> dict[str, object]:
        concept = self.concepts[int(idx)]
        image = _render_image(*concept, image_size=int(self.cfg.image_size))

        gen = torch.Generator().manual_seed(int(self.cfg.seed) * 997 + int(idx))
        negatives = [candidate for candidate in self.all_concepts if candidate != concept]
        perm = torch.randperm(len(negatives), generator=gen).tolist()
        k = int(self.cfg.num_candidates) - 1
        picked = [negatives[i] for i in perm[:k]]
        candidates = [concept, *picked]

        shuffle_order = torch.randperm(len(candidates), generator=gen).tolist()
        candidates = [candidates[i] for i in shuffle_order]
        label_index = int(shuffle_order.index(0))

        candidate_ids: list[torch.Tensor] = []
        candidate_mask: list[torch.Tensor] = []
        candidate_texts: list[str] = []
        for cand in candidates:
            tokens = _caption_tokens(cand)
            ids, mask = self.vocab.encode(tokens, max_length=int(self.cfg.max_text_length))
            candidate_ids.append(ids)
            candidate_mask.append(mask)
            candidate_texts.append(" ".join(tokens))

        return {
            "image": image,
            "candidate_input_ids": torch.stack(candidate_ids, dim=0),
            "candidate_attention_mask": torch.stack(candidate_mask, dim=0),
            "label_index": torch.tensor(label_index, dtype=torch.long),
            "query_text": " ".join(_caption_tokens(concept)),
            "candidate_texts": candidate_texts,
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Vocab]:
    vocab = _build_vocab()
    dataset = SyntheticImageTextRerankingDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "image": torch.stack([sample["image"] for sample in batch], dim=0),
            "candidate_input_ids": torch.stack(
                [sample["candidate_input_ids"] for sample in batch], dim=0
            ),
            "candidate_attention_mask": torch.stack(
                [sample["candidate_attention_mask"] for sample in batch], dim=0
            ),
            "label_index": torch.stack([sample["label_index"] for sample in batch], dim=0),
            "query_text": [str(sample["query_text"]) for sample in batch],
            "candidate_texts": [list(sample["candidate_texts"]) for sample in batch],
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
