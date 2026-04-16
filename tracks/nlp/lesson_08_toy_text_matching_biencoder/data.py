from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices
from tracks.nlp.toy_text import Vocab, simple_tokenize


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 1024
    batch_size: int = 64
    max_query_length: int = 12
    max_doc_length: int = 20
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


_CONCEPTS = (
    {
        "entity": "apple",
        "category": "fruit",
        "attr": "red",
        "extra": "sweet",
        "domain": "market",
    },
    {
        "entity": "banana",
        "category": "fruit",
        "attr": "yellow",
        "extra": "soft",
        "domain": "basket",
    },
    {
        "entity": "car",
        "category": "vehicle",
        "attr": "fast",
        "extra": "road",
        "domain": "garage",
    },
    {
        "entity": "train",
        "category": "vehicle",
        "attr": "long",
        "extra": "rail",
        "domain": "station",
    },
    {
        "entity": "whale",
        "category": "animal",
        "attr": "blue",
        "extra": "ocean",
        "domain": "sea",
    },
    {
        "entity": "eagle",
        "category": "animal",
        "attr": "sharp",
        "extra": "sky",
        "domain": "cliff",
    },
    {
        "entity": "piano",
        "category": "instrument",
        "attr": "black",
        "extra": "melody",
        "domain": "stage",
    },
    {
        "entity": "violin",
        "category": "instrument",
        "attr": "wooden",
        "extra": "orchestra",
        "domain": "hall",
    },
)

_QUERY_TEMPLATES = (
    "find the {attr} {category}",
    "match the {entity} description",
    "retrieve the {category} from the {domain}",
    "which document mentions {extra} {entity}",
)

_DOC_TEMPLATES = (
    "the {entity} is a {attr} {category} often seen in the {domain}",
    "{entity} belongs to the {category} group and is known for being {attr} and {extra}",
    "people describe the {entity} as a {attr} {category} linked with {extra}",
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


def _encode(vocab: Vocab, text: str, *, max_length: int) -> tuple[list[int], list[int]]:
    tokens = simple_tokenize(text)
    ids = [vocab.token_to_id.get(token, vocab.unk_id) for token in tokens[: int(max_length)]]
    mask = [1] * len(ids)
    while len(ids) < int(max_length):
        ids.append(vocab.pad_id)
        mask.append(0)
    return ids, mask


def _make_pairs(num_samples: int, seed: int) -> list[tuple[str, str]]:
    rng = np.random.default_rng(int(seed))
    pairs: list[tuple[str, str]] = []
    for _ in range(int(num_samples)):
        concept = dict(_CONCEPTS[int(rng.integers(0, len(_CONCEPTS)))])
        query = str(rng.choice(_QUERY_TEMPLATES)).format(**concept)
        doc = str(rng.choice(_DOC_TEMPLATES)).format(**concept)
        pairs.append((query, doc))
    return pairs


class ToyTextMatchingDataset:
    def __init__(
        self,
        *,
        pairs: list[tuple[str, str]],
        vocab: Vocab,
        max_query_length: int,
        max_doc_length: int,
    ) -> None:
        self.pairs = list(pairs)
        self.vocab = vocab
        self.max_query_length = int(max_query_length)
        self.max_doc_length = int(max_doc_length)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        import torch

        query, doc = self.pairs[int(idx)]
        query_ids, query_mask = _encode(
            self.vocab,
            query,
            max_length=self.max_query_length,
        )
        doc_ids, doc_mask = _encode(
            self.vocab,
            doc,
            max_length=self.max_doc_length,
        )
        inputs = {
            "query_input_ids": torch.tensor(query_ids, dtype=torch.long),
            "query_attention_mask": torch.tensor(query_mask, dtype=torch.float32),
            "doc_input_ids": torch.tensor(doc_ids, dtype=torch.long),
            "doc_attention_mask": torch.tensor(doc_mask, dtype=torch.float32),
        }
        return inputs, 1


def get_dataloaders(config: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    pairs = _make_pairs(config.num_samples, config.seed)
    vocab = _build_vocab([text for pair in pairs for text in pair])
    dataset = ToyTextMatchingDataset(
        pairs=pairs,
        vocab=vocab,
        max_query_length=config.max_query_length,
        max_doc_length=config.max_doc_length,
    )

    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(config.val_fraction),
        seed=int(config.seed),
    )
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    def _collate(batch):
        query_input_ids = torch.stack([item[0]["query_input_ids"] for item in batch], dim=0)
        query_attention_mask = torch.stack(
            [item[0]["query_attention_mask"] for item in batch],
            dim=0,
        )
        doc_input_ids = torch.stack([item[0]["doc_input_ids"] for item in batch], dim=0)
        doc_attention_mask = torch.stack([item[0]["doc_attention_mask"] for item in batch], dim=0)
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return {
            "query_input_ids": query_input_ids,
            "query_attention_mask": query_attention_mask,
            "doc_input_ids": doc_input_ids,
            "doc_attention_mask": doc_attention_mask,
        }, labels

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


__all__ = ["DataConfig", "ToyTextMatchingDataset", "Vocab", "get_dataloaders"]
