from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def simple_tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


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
        ids = [self.token_to_id.get(t, self.unk_id) for t in tokens][: int(max_length)]
        attn = [1] * len(ids)
        while len(ids) < int(max_length):
            ids.append(self.pad_id)
            attn.append(0)
        return ids, attn

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
    max_length: int = 32
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _make_toy_examples(num_samples: int, seed: int) -> list[tuple[str, int]]:
    rng = np.random.default_rng(int(seed))

    items = ["movie", "book", "game", "meal", "song"]
    pos_adj = ["great", "amazing", "fun", "excellent", "awesome"]
    neg_adj = ["bad", "boring", "terrible", "awful", "poor"]
    pos_verb = ["love", "like", "enjoy", "recommend"]
    neg_verb = ["hate", "dislike", "regret", "avoid"]
    intens = ["really", "very", "quite", "super"]

    pos_templates = [
        "i {verb} this {item}",
        "this {item} is {adj}",
        "what a {adj} {item}",
        "{intens} {adj} {item}",
    ]
    neg_templates = [
        "i {verb} this {item}",
        "this {item} is {adj}",
        "what a {adj} {item}",
        "{intens} {adj} {item}",
    ]

    num_samples = int(num_samples)
    num_pos = num_samples // 2
    num_neg = num_samples - num_pos

    examples: list[tuple[str, int]] = []
    for _ in range(num_pos):
        text = rng.choice(pos_templates).format(
            verb=rng.choice(pos_verb),
            item=rng.choice(items),
            adj=rng.choice(pos_adj),
            intens=rng.choice(intens),
        )
        examples.append((text, 1))
    for _ in range(num_neg):
        text = rng.choice(neg_templates).format(
            verb=rng.choice(neg_verb),
            item=rng.choice(items),
            adj=rng.choice(neg_adj),
            intens=rng.choice(intens),
        )
        examples.append((text, 0))

    rng.shuffle(examples)
    return examples


def _build_vocab(texts: list[str]) -> Vocab:
    tokens: list[str] = []
    for t in texts:
        tokens.extend(simple_tokenize(t))

    # Special tokens first.
    id_to_token = ["<pad>", "<unk>"]
    token_to_id = {"<pad>": 0, "<unk>": 1}

    for tok in sorted(set(tokens)):
        if tok in token_to_id:
            continue
        token_to_id[tok] = len(id_to_token)
        id_to_token.append(tok)

    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token, pad_id=0, unk_id=1)


class ToyTextDataset:
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
        inputs = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.float32),
        }
        return inputs, int(label)


def get_dataloaders(config: DataConfig):
    """Return `(train_loader, val_loader, vocab)` for the toy text task."""

    import torch
    from torch.utils.data import DataLoader, Subset

    examples = _make_toy_examples(num_samples=config.num_samples, seed=config.seed)
    vocab = _build_vocab([t for t, _ in examples])
    ds = ToyTextDataset(examples=examples, vocab=vocab, max_length=config.max_length)

    train_idx, val_idx = train_val_split_indices(
        n=len(ds), val_fraction=float(config.val_fraction), seed=int(config.seed)
    )
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    def _collate(batch):
        input_ids = torch.stack([b[0]["input_ids"] for b in batch], dim=0)
        attention_mask = torch.stack([b[0]["attention_mask"] for b in batch], dim=0)
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}, labels

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        collate_fn=_collate,
    )
    return train_loader, val_loader, vocab


__all__ = ["DataConfig", "Vocab", "ToyTextDataset", "get_dataloaders", "simple_tokenize"]

