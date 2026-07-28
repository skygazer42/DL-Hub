from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices
from tracks.nlp.synthetic_text import Vocab, simple_tokenize


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 256
    batch_size: int = 16
    max_length: int = 14
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


_SCENARIOS = (
    ("cat", "on", "mat"),
    ("dog", "near", "sofa"),
    ("child", "inside", "school"),
    ("chef", "in", "kitchen"),
    ("runner", "on", "track"),
    ("bird", "above", "tree"),
)

_LABELS = ("entailment", "contradiction", "neutral")
_LABEL_TO_ID = {label: idx for idx, label in enumerate(_LABELS)}


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


def _encode(vocab: Vocab, text: str, *, max_length: int) -> tuple[list[int], list[int]]:
    ids = [vocab.token_to_id.get(tok, vocab.unk_id) for tok in simple_tokenize(text)[: int(max_length)]]
    attention = [1] * len(ids)
    while len(ids) < int(max_length):
        ids.append(vocab.pad_id)
        attention.append(0)
    return ids, attention


def _make_pair(subject: str, relation: str, obj: str, label: str, rng: np.random.Generator) -> tuple[str, int]:
    premise = f"premise {subject} is {relation} the {obj}"
    if label == "entailment":
        hypothesis = f"hypothesis {subject} is {relation} the {obj}"
    elif label == "contradiction":
        other_obj = str(rng.choice([x[2] for x in _SCENARIOS if x[2] != obj]))
        hypothesis = f"hypothesis {subject} is {relation} the {other_obj}"
    else:
        hypothesis = f"hypothesis someone mentions {obj}"
    text = f"{premise} {hypothesis} label {label}"
    return text, int(_LABEL_TO_ID[label])


def _make_examples(config: DataConfig) -> list[tuple[str, int]]:
    rng = np.random.default_rng(int(config.seed))
    examples: list[tuple[str, int]] = []
    for _ in range(int(config.num_samples)):
        subject, relation, obj = _SCENARIOS[int(rng.integers(0, len(_SCENARIOS)))]
        label = _LABELS[int(rng.integers(0, len(_LABELS)))]
        examples.append(_make_pair(subject, relation, obj, label, rng))
    rng.shuffle(examples)
    return examples


class TextualEntailmentDataset:
    def __init__(self, *, examples: list[tuple[str, int]], vocab: Vocab, cfg: DataConfig) -> None:
        self.examples = list(examples)
        self.vocab = vocab
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        import torch

        text, label = self.examples[int(idx)]
        input_ids, attention_mask = _encode(self.vocab, text, max_length=int(self.cfg.max_length))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def get_dataloaders(config: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    examples = _make_examples(config)
    vocab = _build_vocab([text for text, _ in examples])
    dataset = TextualEntailmentDataset(examples=examples, vocab=vocab, cfg=config)

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


__all__ = ["DataConfig", "TextualEntailmentDataset", "get_dataloaders"]
