from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

_COLORS: dict[str, tuple[float, float, float]] = {
    "red": (1.0, 0.18, 0.18),
    "blue": (0.2, 0.35, 1.0),
}
_MATERIALS: tuple[str, ...] = ("metal", "wood")


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


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 512
    batch_size: int = 16
    image_size: int = 16
    max_facts_length: int = 12
    max_query_length: int = 8
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "candidate_a",
        "candidate_b",
        "material",
        "metal",
        "wood",
        "is",
        "the",
        "image",
        "object",
        "yes",
        "no",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _render_color_patch(color_name: str, image_size: int) -> torch.Tensor:
    image = torch.full((3, int(image_size), int(image_size)), 0.04, dtype=torch.float32)
    color = torch.tensor(_COLORS[color_name], dtype=torch.float32).view(3, 1, 1)
    inset = max(2, int(image_size) // 4)
    image[:, inset:-inset, inset:-inset] = color
    image[:, 0, :] = 0.08
    image[:, -1, :] = 0.08
    image[:, :, 0] = 0.08
    image[:, :, -1] = 0.08
    return image.clamp(0.0, 1.0)


class SyntheticMultimodalReasoningDataset(Dataset):
    def __init__(self, cfg: DataConfig, vocab: Vocab) -> None:
        self.cfg = cfg
        self.vocab = vocab
        self.generator = torch.Generator().manual_seed(int(cfg.seed))
        if int(cfg.num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        if int(cfg.image_size) < 8:
            raise ValueError("image_size must be >= 8")
        if int(cfg.max_facts_length) < 6:
            raise ValueError("max_facts_length must be >= 6")
        if int(cfg.max_query_length) < 5:
            raise ValueError("max_query_length must be >= 5")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        del idx
        image_color = "red" if bool(torch.randint(2, (1,), generator=self.generator).item() == 0) else "blue"
        query_material = _MATERIALS[int(torch.randint(2, (1,), generator=self.generator).item())]

        candidate_a_material = _MATERIALS[int(torch.randint(2, (1,), generator=self.generator).item())]
        candidate_b_material = _MATERIALS[int(torch.randint(2, (1,), generator=self.generator).item())]

        selected_candidate_material = candidate_a_material if image_color == "red" else candidate_b_material
        label = 1 if selected_candidate_material == query_material else 0

        facts_tokens = [
            "candidate_a",
            "material",
            candidate_a_material,
            "candidate_b",
            "material",
            candidate_b_material,
        ]
        query_tokens = ["is", "the", "image", "object", query_material]
        facts_ids, facts_mask = self.vocab.encode_tokens(
            facts_tokens, max_length=int(self.cfg.max_facts_length)
        )
        query_ids, query_mask = self.vocab.encode_tokens(
            query_tokens, max_length=int(self.cfg.max_query_length)
        )

        return {
            "image": _render_color_patch(image_color, int(self.cfg.image_size)),
            "facts_ids": facts_ids,
            "facts_mask": facts_mask,
            "query_ids": query_ids,
            "query_mask": query_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "answer_text": "yes" if label == 1 else "no",
            "query_text": " ".join(query_tokens),
            "facts_text": " ".join(facts_tokens),
            "image_color": image_color,
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Vocab]:
    vocab = _build_vocab()
    dataset = SyntheticMultimodalReasoningDataset(cfg, vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "image": torch.stack([sample["image"] for sample in batch], dim=0),
            "facts_ids": torch.stack([sample["facts_ids"] for sample in batch], dim=0),
            "facts_mask": torch.stack([sample["facts_mask"] for sample in batch], dim=0),
            "query_ids": torch.stack([sample["query_ids"] for sample in batch], dim=0),
            "query_mask": torch.stack([sample["query_mask"] for sample in batch], dim=0),
            "labels": torch.stack([sample["labels"] for sample in batch], dim=0),
            "answer_text": [str(sample["answer_text"]) for sample in batch],
            "query_text": [str(sample["query_text"]) for sample in batch],
            "facts_text": [str(sample["facts_text"]) for sample in batch],
            "image_color": [str(sample["image_color"]) for sample in batch],
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
