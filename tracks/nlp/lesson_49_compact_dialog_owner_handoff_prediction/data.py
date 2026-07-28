from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices
from tracks.nlp.synthetic_text import Vocab, simple_tokenize


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 320
    batch_size: int = 16
    max_length: int = 28
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


HANDOFF_LABELS = ("none", "billing", "support", "operations")
HANDOFF_TO_ID = {name: idx for idx, name in enumerate(HANDOFF_LABELS)}

GENERAL_ISSUES = ("invoice", "refund", "password", "outage", "routing", "sync")
CHANNELS = ("dialog", "chat", "portal")
MOODS = ("calm", "frustrated", "urgent")
STAGES = ("triage", "investigation", "closure")
SIGNALS = ("sla", "policy", "device", "queue")


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


def _build_example(label: str, rng: np.random.Generator) -> tuple[str, int]:
    channel = str(rng.choice(CHANNELS))
    mood = str(rng.choice(MOODS))
    stage = str(rng.choice(STAGES))
    signal = str(rng.choice(SIGNALS))
    issue = str(rng.choice(GENERAL_ISSUES))
    age_hours = int(rng.integers(1, 48))

    if label == "none":
        text = (
            f"dialog {channel} issue {issue} mood {mood} stage {stage} "
            f"owner keeps queue no handoff needed closure ready after {age_hours} hours signal {signal}"
        )
    else:
        text = (
            f"dialog {channel} issue {issue} mood {mood} stage {stage} "
            f"handoff to {label} owner queue before closure after {age_hours} hours signal {signal}"
        )
    return text, int(HANDOFF_TO_ID[label])


def _make_examples(config: DataConfig) -> list[tuple[str, int]]:
    rng = np.random.default_rng(int(config.seed))
    examples: list[tuple[str, int]] = []
    for _ in range(int(config.num_samples)):
        label = str(rng.choice(HANDOFF_LABELS))
        examples.append(_build_example(label, rng))
    rng.shuffle(examples)
    return examples


class DialogOwnerHandoffDataset:
    def __init__(self, *, examples: list[tuple[str, int]], vocab: Vocab, max_length: int) -> None:
        self.examples = list(examples)
        self.vocab = vocab
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        import torch

        text, label = self.examples[int(idx)]
        ids, attn = self.vocab.encode(text, max_length=self.max_length)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.float32),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def get_dataloaders(config: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    examples = _make_examples(config)
    vocab = _build_vocab([text for text, _ in examples])
    dataset = DialogOwnerHandoffDataset(examples=examples, vocab=vocab, max_length=int(config.max_length))

    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(config.val_fraction),
        seed=int(config.seed),
    )

    def _collate(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
            "labels": torch.stack([item["labels"] for item in batch], dim=0),
        }

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        collate_fn=_collate,
    )
    return train_loader, val_loader, vocab


__all__ = ["DataConfig", "DialogOwnerHandoffDataset", "get_dataloaders"]
