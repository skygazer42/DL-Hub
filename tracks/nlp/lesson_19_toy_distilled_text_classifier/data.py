from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices
from tracks.nlp.toy_text import Vocab, simple_tokenize


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 64
    max_length: int = 16
    num_classes: int = 4
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


_CLASSES = (
    {"label": 0, "name": "weather", "keyword": "rain", "context": "forecast", "action": "umbrella"},
    {"label": 1, "name": "navigation", "keyword": "route", "context": "traffic", "action": "map"},
    {"label": 2, "name": "music", "keyword": "playlist", "context": "melody", "action": "speaker"},
    {"label": 3, "name": "finance", "keyword": "budget", "context": "expense", "action": "report"},
)

_TEMPLATES = (
    "{name} request uses {keyword} with {context}",
    "user asks for {action} when {keyword} appears",
    "{context} update keeps {keyword} near the {action}",
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


def _make_examples(num_samples: int, num_classes: int, seed: int) -> list[tuple[str, int]]:
    rng = np.random.default_rng(int(seed))
    classes = _CLASSES[: int(num_classes)]
    examples: list[tuple[str, int]] = []
    for _ in range(int(num_samples)):
        concept = dict(classes[int(rng.integers(0, len(classes)))])
        template = str(rng.choice(_TEMPLATES))
        examples.append((template.format(**concept), int(concept["label"])))
    rng.shuffle(examples)
    return examples


class DistilledTextClassifierDataset:
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
            "labels": torch.tensor(label, dtype=torch.long),
        }


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    examples = _make_examples(int(cfg.num_samples), int(cfg.num_classes), int(cfg.seed))
    vocab = _build_vocab([text for text, _ in examples])
    dataset = DistilledTextClassifierDataset(examples=examples, vocab=vocab, max_length=int(cfg.max_length))

    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    def _collate(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
            "labels": torch.stack([item["labels"] for item in batch], dim=0),
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


__all__ = ["DataConfig", "DistilledTextClassifierDataset", "get_dataloaders"]

