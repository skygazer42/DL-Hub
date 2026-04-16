from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices
from tracks.nlp.toy_text import Vocab, simple_tokenize

CUISINES = ("none", "italian", "sushi", "indian")
AREAS = ("none", "downtown", "uptown", "riverside")
PARTIES = ("none", "two", "four", "six")

@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 320
    batch_size: int = 16
    max_length: int = 24
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


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


def _sample_slot_value(rng: np.random.Generator, values: tuple[str, ...]) -> tuple[str, int]:
    idx = int(rng.integers(0, len(values)))
    return values[idx], idx


def _generate_example(rng: np.random.Generator) -> tuple[str, tuple[int, int, int]]:
    cuisine_value, cuisine_label = _sample_slot_value(rng, CUISINES)
    area_value, area_label = _sample_slot_value(rng, AREAS)
    party_value, party_label = _sample_slot_value(rng, PARTIES)

    turns = [
        "user request slot filling for restaurant booking",
        f"user slot cuisine {cuisine_value}",
        f"user slot area {area_value}",
        f"user slot party {party_value}",
        "system confirm slot values before action",
    ]
    rng.shuffle(turns[1:4])
    text = " ".join(turns)
    return text, (cuisine_label, area_label, party_label)


def _make_examples(config: DataConfig) -> list[tuple[str, tuple[int, int, int]]]:
    rng = np.random.default_rng(int(config.seed))
    examples = [_generate_example(rng) for _ in range(int(config.num_samples))]
    rng.shuffle(examples)
    return examples


class DialogSlotDataset:
    def __init__(self, *, examples: list[tuple[str, tuple[int, int, int]]], vocab: Vocab, max_length: int) -> None:
        self.examples = list(examples)
        self.vocab = vocab
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        import torch

        text, (cuisine_label, area_label, party_label) = self.examples[int(idx)]
        input_ids, attention_mask = self.vocab.encode(text, max_length=self.max_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "cuisine_labels": torch.tensor(int(cuisine_label), dtype=torch.long),
            "area_labels": torch.tensor(int(area_label), dtype=torch.long),
            "party_labels": torch.tensor(int(party_label), dtype=torch.long),
        }


def get_dataloaders(config: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    examples = _make_examples(config)
    vocab = _build_vocab([text for text, _ in examples])
    dataset = DialogSlotDataset(examples=examples, vocab=vocab, max_length=int(config.max_length))

    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(config.val_fraction),
        seed=int(config.seed),
    )

    def _collate(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
            "cuisine_labels": torch.stack([item["cuisine_labels"] for item in batch], dim=0),
            "area_labels": torch.stack([item["area_labels"] for item in batch], dim=0),
            "party_labels": torch.stack([item["party_labels"] for item in batch], dim=0),
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


__all__ = ["AREAS", "CUISINES", "PARTIES", "DataConfig", "DialogSlotDataset", "get_dataloaders"]
