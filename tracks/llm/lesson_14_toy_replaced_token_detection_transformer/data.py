from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class Vocab:
    pad_id: int = 0
    prompt_token_id: int = 1
    separator_token_id: int = 2
    eos_id: int = 3
    base_vocab_size: int = 64
    ignore_index: int = -100

    @property
    def content_start_id(self) -> int:
        return 4

    @property
    def size(self) -> int:
        return int(self.content_start_id + self.base_vocab_size)

    def to_dict(self) -> dict[str, int]:
        return {
            "pad_id": int(self.pad_id),
            "prompt_token_id": int(self.prompt_token_id),
            "separator_token_id": int(self.separator_token_id),
            "eos_id": int(self.eos_id),
            "base_vocab_size": int(self.base_vocab_size),
            "ignore_index": int(self.ignore_index),
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 4096
    batch_size: int = 64
    seq_length: int = 18
    base_vocab_size: int = 64
    replace_probability: float = 0.25
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _next_content_token(token_id: int, vocab: Vocab) -> int:
    base = int(token_id - vocab.content_start_id)
    return int(vocab.content_start_id + ((base + 1) % int(vocab.base_vocab_size)))


def _build_clean_sample(topic_id: int, vocab: Vocab, seq_length: int) -> np.ndarray:
    c1 = _next_content_token(topic_id, vocab)
    c2 = _next_content_token(c1, vocab)
    c3 = _next_content_token(c2, vocab)
    tokens = [
        int(vocab.prompt_token_id),
        int(topic_id),
        int(vocab.separator_token_id),
        int(topic_id),
        int(c1),
        int(c2),
        int(c3),
        int(vocab.eos_id),
    ]
    input_ids = np.full((seq_length,), fill_value=int(vocab.pad_id), dtype=np.int64)
    used = min(len(tokens), seq_length)
    input_ids[:used] = np.asarray(tokens[:used], dtype=np.int64)
    return input_ids


def _corrupt_tokens(
    clean_ids: np.ndarray,
    *,
    vocab: Vocab,
    replace_probability: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    corrupted = clean_ids.copy()
    replaced = np.zeros_like(clean_ids, dtype=np.float32)
    special = {
        int(vocab.pad_id),
        int(vocab.prompt_token_id),
        int(vocab.separator_token_id),
        int(vocab.eos_id),
    }
    for pos in range(int(clean_ids.shape[0])):
        token = int(clean_ids[pos])
        if token in special:
            continue
        if float(rng.random()) >= float(replace_probability):
            continue
        candidate = int(vocab.content_start_id + int(rng.integers(0, int(vocab.base_vocab_size))))
        if candidate == token:
            candidate = _next_content_token(token, vocab)
        corrupted[pos] = int(candidate)
        replaced[pos] = 1.0
    return corrupted, replaced


def _make_dataset_arrays(cfg: DataConfig, vocab: Vocab) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(cfg.seed))
    n = int(cfg.num_samples)
    t = int(cfg.seq_length)

    input_ids = np.empty((n, t), dtype=np.int64)
    labels = np.empty((n, t), dtype=np.int64)
    replaced_labels = np.empty((n, t), dtype=np.float32)

    for i in range(n):
        topic_offset = int(rng.integers(0, int(vocab.base_vocab_size)))
        topic_id = int(vocab.content_start_id + topic_offset)
        clean = _build_clean_sample(topic_id, vocab, t)
        corrupted, replaced = _corrupt_tokens(
            clean,
            vocab=vocab,
            replace_probability=float(cfg.replace_probability),
            rng=rng,
        )
        input_ids[i] = corrupted
        labels[i] = clean
        replaced_labels[i] = replaced
    return input_ids, labels, replaced_labels


class ToyReplacedTokenDetectionDataset:
    def __init__(
        self,
        *,
        input_ids: np.ndarray,
        labels: np.ndarray,
        replaced_labels: np.ndarray,
        vocab: Vocab,
    ) -> None:
        self.input_ids = input_ids
        self.labels = labels
        self.replaced_labels = replaced_labels
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int):
        import torch

        input_ids = torch.tensor(self.input_ids[int(idx)], dtype=torch.long)
        labels = torch.tensor(self.labels[int(idx)], dtype=torch.long)
        replaced_labels = torch.tensor(self.replaced_labels[int(idx)], dtype=torch.float32)
        attention_mask = (input_ids != int(self.vocab.pad_id)).to(torch.float32)
        labels = labels.masked_fill(attention_mask.eq(0), int(self.vocab.ignore_index))
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "replaced_labels": replaced_labels,
        }


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    if int(cfg.seq_length) < 12:
        raise ValueError("seq_length must be >= 12 for toy replaced-token detection sequences")

    vocab = Vocab(base_vocab_size=int(cfg.base_vocab_size))
    arrays = _make_dataset_arrays(cfg, vocab)
    ds = ToyReplacedTokenDetectionDataset(
        input_ids=arrays[0],
        labels=arrays[1],
        replaced_labels=arrays[2],
        vocab=vocab,
    )
    train_idx, val_idx = train_val_split_indices(
        n=len(ds), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    def _collate(batch):
        return {
            "input_ids": torch.stack([row["input_ids"] for row in batch], dim=0),
            "attention_mask": torch.stack([row["attention_mask"] for row in batch], dim=0),
            "labels": torch.stack([row["labels"] for row in batch], dim=0),
            "replaced_labels": torch.stack([row["replaced_labels"] for row in batch], dim=0),
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


__all__ = ["DataConfig", "ToyReplacedTokenDetectionDataset", "Vocab", "get_dataloaders"]
