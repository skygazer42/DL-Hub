from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

_OBJECTS: tuple[str, ...] = ("cup", "book", "phone", "ball")
_RELATIONS: tuple[str, ...] = ("holding", "looking")
_OBJECT_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(_OBJECTS)}
_REL_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(_RELATIONS)}


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

    def encode_tokens(self, tokens: list[str], *, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = [self.bos_id, *[int(self.token_to_id[token]) for token in tokens], self.eos_id]
        if len(seq) > int(max_length):
            raise ValueError(f"Sequence exceeds max_length={int(max_length)}")
        pad_count = int(max_length) - len(seq)
        seq.extend([self.pad_id] * pad_count)
        mask = [1.0] * (int(max_length) - pad_count) + [0.0] * pad_count
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
    batch_size: int = 16
    num_regions: int = 6
    feature_dim: int = 16
    max_query_length: int = 10
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "person",
        "holding",
        "looking",
        "cup",
        "book",
        "phone",
        "ball",
        "yes",
        "no",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _make_boxes(num_regions: int) -> torch.Tensor:
    boxes = torch.zeros(int(num_regions), 4, dtype=torch.float32)
    for idx in range(int(num_regions)):
        x0 = 0.05 + 0.12 * float(idx % 3)
        y0 = 0.08 + 0.16 * float(idx // 3)
        x1 = min(0.95, x0 + 0.25)
        y1 = min(0.95, y0 + 0.25)
        boxes[idx] = torch.tensor([x0, y0, x1, y1], dtype=torch.float32)
    return boxes


def _region_features(
    *,
    cfg: DataConfig,
    target_object: str,
    target_relation: str,
    is_positive: bool,
    generator: torch.Generator,
) -> torch.Tensor:
    num_regions = int(cfg.num_regions)
    feature_dim = int(cfg.feature_dim)
    regions = torch.zeros(num_regions, feature_dim, dtype=torch.float32)
    regions += 0.02 * torch.randn(num_regions, feature_dim, generator=generator, dtype=torch.float32)

    rel_id = int(_REL_TO_ID[target_relation])
    obj_id = int(_OBJECT_TO_ID[target_object])

    matched_object = target_object if is_positive else _OBJECTS[(obj_id + 1) % len(_OBJECTS)]
    matched_obj_id = int(_OBJECT_TO_ID[matched_object])
    matched_relation = target_relation if is_positive else _RELATIONS[(rel_id + 1) % len(_RELATIONS)]

    # Region 0 is the person anchor with encoded interaction signal.
    regions[0, 0] = 1.0
    regions[0, 2 + int(_REL_TO_ID[matched_relation])] = 1.0
    regions[0, 4 + matched_obj_id] = 1.0

    # Region 1 is the queried object.
    regions[1, 1] = 1.0
    regions[1, 4 + obj_id] = 1.0

    # Remaining regions are distractor objects.
    for idx in range(2, num_regions):
        distractor_obj = int(torch.randint(0, len(_OBJECTS), (1,), generator=generator).item())
        regions[idx, 1] = 1.0
        regions[idx, 4 + distractor_obj] = 0.8

    return regions


class SyntheticHoiReasoningDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        self.cfg = cfg
        self.vocab = vocab
        if int(cfg.num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        if int(cfg.num_regions) < 3:
            raise ValueError("num_regions must be >= 3")
        if int(cfg.feature_dim) < 10:
            raise ValueError("feature_dim must be >= 10")
        if int(cfg.max_query_length) < 5:
            raise ValueError("max_query_length must be >= 5")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        generator = torch.Generator().manual_seed(int(self.cfg.seed) * 997 + int(idx))
        target_object = _OBJECTS[int(torch.randint(0, len(_OBJECTS), (1,), generator=generator).item())]
        target_relation = _RELATIONS[int(torch.randint(0, len(_RELATIONS), (1,), generator=generator).item())]
        is_positive = bool(torch.randint(0, 2, (1,), generator=generator).item() == 1)
        label = 1 if is_positive else 0

        query_tokens = ["person", target_relation, target_object]
        query_ids, query_mask = self.vocab.encode_tokens(
            query_tokens, max_length=int(self.cfg.max_query_length)
        )

        return {
            "region_features": _region_features(
                cfg=self.cfg,
                target_object=target_object,
                target_relation=target_relation,
                is_positive=is_positive,
                generator=generator,
            ),
            "region_boxes": _make_boxes(int(self.cfg.num_regions)),
            "query_ids": query_ids,
            "query_mask": query_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "query_text": " ".join(query_tokens),
            "answer_text": "yes" if label == 1 else "no",
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Vocab]:
    vocab = _build_vocab()
    dataset = SyntheticHoiReasoningDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "region_features": torch.stack([sample["region_features"] for sample in batch], dim=0),
            "region_boxes": torch.stack([sample["region_boxes"] for sample in batch], dim=0),
            "query_ids": torch.stack([sample["query_ids"] for sample in batch], dim=0),
            "query_mask": torch.stack([sample["query_mask"] for sample in batch], dim=0),
            "labels": torch.stack([sample["labels"] for sample in batch], dim=0),
            "query_text": [str(sample["query_text"]) for sample in batch],
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


__all__ = ["DataConfig", "SyntheticHoiReasoningDataset", "Vocab", "get_dataloaders"]
