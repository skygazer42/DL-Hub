from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices
from tracks.nlp.toy_text import simple_tokenize


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
    anomaly_fraction: float = 0.35
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


_NORMAL_TEMPLATES = (
    "the analyst reviews the report in the office",
    "our teammate tests the pipeline in the lab",
    "the student writes notes for the class",
    "the engineer debugs code before release",
    "the scientist studies samples in the lab",
)

_ANOMALY_TEMPLATES = (
    "volcano budget telescope llama sings binary thunder",
    "asteroid recipe circuit glacier dances invoice",
    "mismatch token meadow engine violin cactus",
    "abnormal query mixes orbit pasta kernel museum",
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


def _make_examples(num_samples: int, anomaly_fraction: float, seed: int) -> list[tuple[str, int]]:
    rng = np.random.default_rng(int(seed))
    num_anomaly = int(round(int(num_samples) * float(anomaly_fraction)))
    num_normal = int(num_samples) - num_anomaly
    examples: list[tuple[str, int]] = []
    for _ in range(num_normal):
        examples.append((str(rng.choice(_NORMAL_TEMPLATES)), 0))
    for _ in range(num_anomaly):
        examples.append((str(rng.choice(_ANOMALY_TEMPLATES)), 1))
    rng.shuffle(examples)
    return examples


class TextAnomalyDataset:
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
            "labels": torch.tensor(float(label), dtype=torch.float32),
        }


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    examples = _make_examples(int(cfg.num_samples), float(cfg.anomaly_fraction), int(cfg.seed))
    vocab = _build_vocab([text for text, _ in examples])
    dataset = TextAnomalyDataset(examples=examples, vocab=vocab, max_length=int(cfg.max_length))

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


__all__ = ["DataConfig", "TextAnomalyDataset", "Vocab", "get_dataloaders"]
