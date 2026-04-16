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
    sep_id: int

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def to_dict(self) -> dict[str, object]:
        return {
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "sep_id": self.sep_id,
            "token_to_id": dict(self.token_to_id),
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 64
    max_query_length: int = 8
    max_doc_length: int = 12
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


_CONCEPTS = (
    {"entity": "apple", "category": "fruit", "attr": "red", "extra": "sweet", "domain": "market"},
    {"entity": "banana", "category": "fruit", "attr": "yellow", "extra": "soft", "domain": "basket"},
    {"entity": "car", "category": "vehicle", "attr": "fast", "extra": "road", "domain": "garage"},
    {"entity": "train", "category": "vehicle", "attr": "long", "extra": "rail", "domain": "station"},
    {"entity": "whale", "category": "animal", "attr": "blue", "extra": "ocean", "domain": "sea"},
    {"entity": "eagle", "category": "animal", "attr": "sharp", "extra": "sky", "domain": "cliff"},
    {"entity": "piano", "category": "instrument", "attr": "black", "extra": "melody", "domain": "stage"},
    {"entity": "violin", "category": "instrument", "attr": "wooden", "extra": "orchestra", "domain": "hall"},
)

_QUERY_TEMPLATES = (
    "find the {attr} {category}",
    "match the {entity} description",
    "which item mentions {extra}",
    "retrieve the {category} from the {domain}",
)

_POSITIVE_DOC_TEMPLATES = (
    "the {entity} is a {attr} {category} often seen in the {domain}",
    "{entity} belongs to the {category} group and is known for being {attr} and {extra}",
    "people describe the {entity} as a {attr} {category} linked with {extra}",
)

_NEGATIVE_DOC_TEMPLATES = (
    "the {entity} is unrelated to that query and belongs near the {domain}",
    "{entity} is another {category} but its details center on {extra}",
    "this passage mentions {entity} and {extra} instead of the requested concept",
)


def _build_vocab(texts: list[str]) -> Vocab:
    token_to_id = {"<pad>": 0, "<unk>": 1, "<sep>": 2}
    id_to_token = ["<pad>", "<unk>", "<sep>"]
    for text in texts:
        for token in simple_tokenize(text):
            if token in token_to_id:
                continue
            token_to_id[token] = len(id_to_token)
            id_to_token.append(token)
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token, pad_id=0, unk_id=1, sep_id=2)


def _encode_text(vocab: Vocab, text: str, *, max_length: int) -> tuple[list[int], list[int]]:
    tokens = simple_tokenize(text)
    ids = [vocab.token_to_id.get(token, vocab.unk_id) for token in tokens[: int(max_length)]]
    mask = [1] * len(ids)
    while len(ids) < int(max_length):
        ids.append(vocab.pad_id)
        mask.append(0)
    return ids, mask


def _encode_pair(
    vocab: Vocab,
    *,
    query: str,
    doc: str,
    max_query_length: int,
    max_doc_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    query_ids, query_mask = _encode_text(vocab, query, max_length=max_query_length)
    doc_ids, doc_mask = _encode_text(vocab, doc, max_length=max_doc_length)
    input_ids = np.asarray(query_ids + [vocab.sep_id] + doc_ids, dtype=np.int64)
    attention_mask = np.asarray(query_mask + [1] + doc_mask, dtype=np.float32)
    return input_ids, attention_mask


def _make_triplets(num_samples: int, seed: int) -> list[tuple[str, str, str]]:
    rng = np.random.default_rng(int(seed))
    triplets: list[tuple[str, str, str]] = []
    for _ in range(int(num_samples)):
        pos_idx = int(rng.integers(0, len(_CONCEPTS)))
        neg_choices = [idx for idx in range(len(_CONCEPTS)) if idx != pos_idx]
        neg_idx = int(rng.choice(neg_choices))

        positive = dict(_CONCEPTS[pos_idx])
        negative = dict(_CONCEPTS[neg_idx])
        query = str(rng.choice(_QUERY_TEMPLATES)).format(**positive)
        positive_doc = str(rng.choice(_POSITIVE_DOC_TEMPLATES)).format(**positive)
        negative_doc = str(rng.choice(_NEGATIVE_DOC_TEMPLATES)).format(**negative)
        triplets.append((query, positive_doc, negative_doc))
    return triplets


class ToyCrossEncoderDataset:
    def __init__(
        self,
        *,
        triplets: list[tuple[str, str, str]],
        vocab: Vocab,
        cfg: DataConfig,
    ) -> None:
        self.triplets = list(triplets)
        self.vocab = vocab
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        import torch

        query, positive_doc, negative_doc = self.triplets[int(idx)]
        positive_input_ids, positive_attention_mask = _encode_pair(
            self.vocab,
            query=query,
            doc=positive_doc,
            max_query_length=int(self.cfg.max_query_length),
            max_doc_length=int(self.cfg.max_doc_length),
        )
        negative_input_ids, negative_attention_mask = _encode_pair(
            self.vocab,
            query=query,
            doc=negative_doc,
            max_query_length=int(self.cfg.max_query_length),
            max_doc_length=int(self.cfg.max_doc_length),
        )
        return {
            "positive_input_ids": torch.tensor(positive_input_ids, dtype=torch.long),
            "positive_attention_mask": torch.tensor(positive_attention_mask, dtype=torch.float32),
            "negative_input_ids": torch.tensor(negative_input_ids, dtype=torch.long),
            "negative_attention_mask": torch.tensor(negative_attention_mask, dtype=torch.float32),
        }


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    triplets = _make_triplets(cfg.num_samples, cfg.seed)
    vocab = _build_vocab([text for triplet in triplets for text in triplet])
    dataset = ToyCrossEncoderDataset(triplets=triplets, vocab=vocab, cfg=cfg)

    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    def _collate(batch):
        return {
            "positive_input_ids": torch.stack([item["positive_input_ids"] for item in batch], dim=0),
            "positive_attention_mask": torch.stack(
                [item["positive_attention_mask"] for item in batch],
                dim=0,
            ),
            "negative_input_ids": torch.stack([item["negative_input_ids"] for item in batch], dim=0),
            "negative_attention_mask": torch.stack(
                [item["negative_attention_mask"] for item in batch],
                dim=0,
            ),
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


__all__ = ["DataConfig", "ToyCrossEncoderDataset", "Vocab", "get_dataloaders"]
