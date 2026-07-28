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
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens][: int(max_length)]
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
    num_labeling_functions: int = 3
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab(texts: list[str]) -> Vocab:
    id_to_token = ["<pad>", "<unk>"]
    token_to_id = {"<pad>": 0, "<unk>": 1}
    tokens: list[str] = []
    for text in texts:
        tokens.extend(simple_tokenize(text))
    for token in sorted(set(tokens)):
        if token not in token_to_id:
            token_to_id[token] = len(id_to_token)
            id_to_token.append(token)
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token, pad_id=0, unk_id=1)


def _make_examples(num_samples: int, seed: int) -> list[tuple[str, int]]:
    rng = np.random.default_rng(int(seed))
    items = ["movie", "book", "game", "meal", "song"]
    pos_adj = ["great", "amazing", "fun", "excellent", "awesome"]
    neg_adj = ["bad", "boring", "terrible", "awful", "poor"]
    pos_verb = ["love", "like", "enjoy", "recommend"]
    neg_verb = ["hate", "dislike", "regret", "avoid"]
    hedge = ["maybe", "perhaps", "honestly", "today", "still"]
    templates = [
        "i {verb} this {item}",
        "this {item} is {adj}",
        "{hedge} this {item} feels {adj}",
        "i would {verb} this {item}",
    ]

    examples: list[tuple[str, int]] = []
    for idx in range(int(num_samples)):
        label = 1 if idx % 2 == 0 else 0
        text = rng.choice(templates).format(
            verb=rng.choice(pos_verb if label == 1 else neg_verb),
            item=rng.choice(items),
            adj=rng.choice(pos_adj if label == 1 else neg_adj),
            hedge=rng.choice(hedge),
        )
        examples.append((text, label))
    rng.shuffle(examples)
    return examples


def _labeling_function_votes(
    text: str, gold_label: int, *, num_labeling_functions: int, seed: int
) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(int(seed))
    tokens = set(simple_tokenize(text))
    vote_specs = [
        ({"great", "amazing", "excellent", "awesome"}, {"bad", "terrible", "awful", "poor"}),
        ({"love", "like", "enjoy", "recommend"}, {"hate", "dislike", "regret", "avoid"}),
        ({"fun"}, {"boring"}),
    ]
    votes: list[int] = []
    mask: list[int] = []
    for idx in range(int(num_labeling_functions)):
        pos_words, neg_words = vote_specs[idx % len(vote_specs)]
        vote = -1
        if tokens & pos_words:
            vote = 1
        elif tokens & neg_words:
            vote = 0
        if vote != -1 and rng.random() < 0.15:
            vote = 1 - vote
        if vote != -1 and rng.random() < 0.20:
            vote = -1
        if vote == -1 and rng.random() < 0.35:
            vote = int(gold_label if rng.random() < 0.7 else 1 - gold_label)
        votes.append(0 if vote == -1 else int(vote))
        mask.append(0 if vote == -1 else 1)
    return votes, mask


def _soft_label_from_votes(votes: list[int], mask: list[int]) -> np.ndarray:
    active = np.array(mask, dtype=np.float32)
    votes_arr = np.array(votes, dtype=np.float32)
    pos = float(((votes_arr == 1.0) * active).sum())
    neg = float(((votes_arr == 0.0) * active).sum())
    total = pos + neg
    if total == 0.0:
        return np.array([0.5, 0.5], dtype=np.float32)
    smooth = 0.2
    probs = np.array([neg + smooth, pos + smooth], dtype=np.float32)
    return probs / probs.sum()


class WeakSupervisionTextDataset:
    def __init__(self, *, examples: list[tuple[str, int]], vocab: Vocab, cfg: DataConfig) -> None:
        self.examples = list(examples)
        self.vocab = vocab
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        import torch

        text, gold_label = self.examples[int(idx)]
        input_ids, attention_mask = self.vocab.encode(text, max_length=int(self.cfg.max_length))
        lf_votes, lf_mask = _labeling_function_votes(
            text,
            int(gold_label),
            num_labeling_functions=int(self.cfg.num_labeling_functions),
            seed=int(self.cfg.seed) * 1543 + int(idx),
        )
        label_probs = _soft_label_from_votes(lf_votes, lf_mask)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "lf_votes": torch.tensor(lf_votes, dtype=torch.long),
            "lf_mask": torch.tensor(lf_mask, dtype=torch.float32),
            "label_probs": torch.tensor(label_probs, dtype=torch.float32),
            "gold_labels": torch.tensor(int(gold_label), dtype=torch.long),
        }


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    examples = _make_examples(num_samples=int(cfg.num_samples), seed=int(cfg.seed))
    vocab = _build_vocab([text for text, _ in examples])
    dataset = WeakSupervisionTextDataset(examples=examples, vocab=vocab, cfg=cfg)

    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch):
        keys = batch[0].keys()
        return {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}

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


__all__ = ["DataConfig", "Vocab", "get_dataloaders"]
