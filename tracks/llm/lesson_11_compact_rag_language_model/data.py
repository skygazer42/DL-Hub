from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class Vocab:
    pad_id: int = 0
    query_token_id: int = 1
    doc_token_id: int = 2
    separator_token_id: int = 3
    eos_id: int = 4
    base_vocab_size: int = 64
    num_docs: int = 8
    ignore_index: int = -100

    @property
    def content_start_id(self) -> int:
        return 5

    @property
    def size(self) -> int:
        return int(self.content_start_id + self.base_vocab_size)

    def to_dict(self) -> dict[str, int]:
        return {
            "pad_id": int(self.pad_id),
            "query_token_id": int(self.query_token_id),
            "doc_token_id": int(self.doc_token_id),
            "separator_token_id": int(self.separator_token_id),
            "eos_id": int(self.eos_id),
            "base_vocab_size": int(self.base_vocab_size),
            "num_docs": int(self.num_docs),
            "ignore_index": int(self.ignore_index),
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 4096
    batch_size: int = 64
    seq_length: int = 24
    base_vocab_size: int = 64
    num_docs: int = 8
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _next_content_token(token_id: int, vocab: Vocab) -> int:
    base = int(token_id - vocab.content_start_id)
    nxt = (base + 1) % int(vocab.base_vocab_size)
    return int(vocab.content_start_id + nxt)


def _doc_content_token(doc_id: int, vocab: Vocab) -> int:
    return int(vocab.content_start_id + (int(doc_id) % int(vocab.base_vocab_size)))


def _build_sample(*, topic_id: int, doc_id: int, vocab: Vocab, seq_length: int) -> tuple[np.ndarray, ...]:
    query_tokens = [int(vocab.query_token_id), int(topic_id), int(vocab.separator_token_id)]
    response_tokens = [
        int(vocab.doc_token_id),
        _doc_content_token(doc_id, vocab),
        int(topic_id),
        _next_content_token(topic_id, vocab),
        int(vocab.eos_id),
    ]
    tokens = query_tokens + response_tokens

    input_ids = np.full((seq_length,), fill_value=int(vocab.pad_id), dtype=np.int64)
    labels = np.full((seq_length,), fill_value=int(vocab.ignore_index), dtype=np.int64)

    used = min(len(tokens), int(seq_length))
    input_ids[:used] = np.asarray(tokens[:used], dtype=np.int64)

    # Standard causal LM objective: predict the next token where available.
    if used > 1:
        labels[: (used - 1)] = input_ids[1:used]
    return input_ids, labels, np.asarray(doc_id, dtype=np.int64)


def _make_dataset_arrays(cfg: DataConfig, vocab: Vocab) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(int(cfg.seed))
    n = int(cfg.num_samples)
    t = int(cfg.seq_length)

    input_ids = np.empty((n, t), dtype=np.int64)
    labels = np.empty((n, t), dtype=np.int64)
    doc_ids = np.empty((n,), dtype=np.int64)

    for idx in range(n):
        topic_offset = int(rng.integers(0, int(vocab.base_vocab_size)))
        topic_id = int(vocab.content_start_id + topic_offset)
        doc_id = int(rng.integers(0, int(vocab.num_docs)))
        ids_i, labels_i, doc_i = _build_sample(
            topic_id=topic_id,
            doc_id=doc_id,
            vocab=vocab,
            seq_length=t,
        )
        input_ids[idx] = ids_i
        labels[idx] = labels_i
        doc_ids[idx] = doc_i

    return input_ids, labels, doc_ids


class SyntheticRagLanguageModelDataset:
    def __init__(self, *, input_ids: np.ndarray, labels: np.ndarray, doc_ids: np.ndarray, vocab: Vocab) -> None:
        self.input_ids = input_ids
        self.labels = labels
        self.doc_ids = doc_ids
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int):
        import torch

        ids = torch.tensor(self.input_ids[int(idx)], dtype=torch.long)
        labels = torch.tensor(self.labels[int(idx)], dtype=torch.long)
        doc_id = torch.tensor(self.doc_ids[int(idx)], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": (ids != int(self.vocab.pad_id)).to(torch.float32),
            "labels": labels,
            "doc_ids": doc_id,
        }


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    if int(cfg.seq_length) < 8:
        raise ValueError("seq_length must be >= 8 for synthetic RAG LM sequences")
    if int(cfg.num_docs) < 2:
        raise ValueError("num_docs must be >= 2")

    vocab = Vocab(base_vocab_size=int(cfg.base_vocab_size), num_docs=int(cfg.num_docs))
    input_ids, labels, doc_ids = _make_dataset_arrays(cfg, vocab)
    ds = SyntheticRagLanguageModelDataset(input_ids=input_ids, labels=labels, doc_ids=doc_ids, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(ds),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    def _collate(batch):
        return {
            "input_ids": torch.stack([row["input_ids"] for row in batch], dim=0),
            "attention_mask": torch.stack([row["attention_mask"] for row in batch], dim=0),
            "labels": torch.stack([row["labels"] for row in batch], dim=0),
            "doc_ids": torch.stack([row["doc_ids"] for row in batch], dim=0),
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


__all__ = ["DataConfig", "SyntheticRagLanguageModelDataset", "Vocab", "get_dataloaders"]
