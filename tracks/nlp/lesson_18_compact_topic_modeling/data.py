from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices
from tracks.nlp.synthetic_text import Vocab, simple_tokenize


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 64
    max_length: int = 16
    num_topics: int = 4
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


_TOPICS = (
    {
        "label": 0,
        "name": "science",
        "subject": "researcher",
        "verb": "studies",
        "object": "molecule",
        "place": "lab",
        "keyword": "experiment",
    },
    {
        "label": 1,
        "name": "sports",
        "subject": "athlete",
        "verb": "trains",
        "object": "relay",
        "place": "stadium",
        "keyword": "fitness",
    },
    {
        "label": 2,
        "name": "finance",
        "subject": "analyst",
        "verb": "reviews",
        "object": "budget",
        "place": "office",
        "keyword": "market",
    },
    {
        "label": 3,
        "name": "travel",
        "subject": "traveler",
        "verb": "plans",
        "object": "itinerary",
        "place": "airport",
        "keyword": "ticket",
    },
)

_TEMPLATES = (
    "{subject} {verb} the {object} in the {place}",
    "{name} topic keeps {keyword} near the {place}",
    "{subject} asks how the {keyword} changes the {object}",
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


def _make_examples(num_samples: int, num_topics: int, seed: int) -> list[tuple[str, int]]:
    rng = np.random.default_rng(int(seed))
    topics = _TOPICS[: int(num_topics)]
    examples: list[tuple[str, int]] = []
    for _ in range(int(num_samples)):
        topic = dict(topics[int(rng.integers(0, len(topics)))])
        template = str(rng.choice(_TEMPLATES))
        examples.append((template.format(**topic), int(topic["label"])))
    return examples


def _encode_bow(vocab: Vocab, text: str) -> list[float]:
    bow = [0.0] * vocab.size
    for token in simple_tokenize(text):
        bow[vocab.token_to_id.get(token, vocab.unk_id)] = 1.0
    return bow


class TopicModelingDataset:
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
            "bow_targets": torch.tensor(_encode_bow(self.vocab, text), dtype=torch.float32),
            "topic_labels": torch.tensor(label, dtype=torch.long),
        }


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    examples = _make_examples(int(cfg.num_samples), int(cfg.num_topics), int(cfg.seed))
    vocab = _build_vocab([text for text, _ in examples])
    dataset = TopicModelingDataset(examples=examples, vocab=vocab, max_length=int(cfg.max_length))

    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    def _collate(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
            "bow_targets": torch.stack([item["bow_targets"] for item in batch], dim=0),
            "topic_labels": torch.stack([item["topic_labels"] for item in batch], dim=0),
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


__all__ = ["DataConfig", "TopicModelingDataset", "get_dataloaders"]

