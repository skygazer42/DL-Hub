from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices
from tracks.nlp.synthetic_text import simple_tokenize


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_id: int
    unk_id: int

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str, *, max_length: int) -> tuple[list[int], list[int]]:
        tokens = simple_tokenize(text)
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens[: int(max_length)]]
        mask = [1] * len(ids)
        while len(ids) < int(max_length):
            ids.append(self.pad_id)
            mask.append(0)
        return ids, mask

    def to_dict(self) -> dict[str, object]:
        return {
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "token_to_id": dict(self.token_to_id),
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 64
    max_length: int = 16
    num_clusters: int = 4
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


_CLUSTERS = (
    {
        "label": 0,
        "name": "science",
        "subject": "researcher",
        "verb": "studies",
        "object": "molecule",
        "place": "lab",
    },
    {
        "label": 1,
        "name": "sports",
        "subject": "athlete",
        "verb": "practices",
        "object": "drill",
        "place": "stadium",
    },
    {
        "label": 2,
        "name": "finance",
        "subject": "analyst",
        "verb": "reviews",
        "object": "budget",
        "place": "office",
    },
    {
        "label": 3,
        "name": "art",
        "subject": "artist",
        "verb": "sketches",
        "object": "poster",
        "place": "studio",
    },
)

_TEMPLATES = (
    "{subject} {verb} the {object} in the {place}",
    "{name} topic keeps the {object} near the {place}",
    "cluster {name} asks how the {subject} will {verb}",
)


def _build_vocab(texts: list[str]) -> Vocab:
    token_to_id = {"<pad>": 0, "<unk>": 1}
    id_to_token = ["<pad>", "<unk>"]
    for text in texts:
        for token in simple_tokenize(text):
            if token in token_to_id:
                continue
            token_to_id[token] = len(id_to_token)
            id_to_token.append(token)
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token, pad_id=0, unk_id=1)


def _make_examples(num_samples: int, num_clusters: int, seed: int) -> list[tuple[str, int]]:
    rng = np.random.default_rng(int(seed))
    clusters = _CLUSTERS[: int(num_clusters)]
    examples: list[tuple[str, int]] = []
    for _ in range(int(num_samples)):
        cluster = dict(clusters[int(rng.integers(0, len(clusters)))])
        template = str(rng.choice(_TEMPLATES))
        examples.append((template.format(**cluster), int(cluster["label"])))
    return examples


class TextClusteringDataset:
    def __init__(self, *, examples: list[tuple[str, int]], vocab: Vocab, max_length: int) -> None:
        self.examples = list(examples)
        self.vocab = vocab
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        import torch

        text, label = self.examples[int(idx)]
        ids, mask = self.vocab.encode(text, max_length=self.max_length)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.float32),
            "cluster_labels": torch.tensor(label, dtype=torch.long),
        }


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    examples = _make_examples(int(cfg.num_samples), int(cfg.num_clusters), int(cfg.seed))
    vocab = _build_vocab([text for text, _ in examples])
    dataset = TextClusteringDataset(examples=examples, vocab=vocab, max_length=int(cfg.max_length))

    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    def _collate(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
            "cluster_labels": torch.stack([item["cluster_labels"] for item in batch], dim=0),
        }

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    return train_loader, val_loader, vocab


__all__ = ["DataConfig", "TextClusteringDataset", "Vocab", "get_dataloaders"]
