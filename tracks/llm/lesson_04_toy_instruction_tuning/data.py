from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class Vocab:
    pad_id: int = 0
    instruction_token_id: int = 1
    input_token_id: int = 2
    response_token_id: int = 3
    eos_id: int = 4
    task_token_id: int = 5
    separator_token_id: int = 6
    base_vocab_size: int = 64
    ignore_index: int = -100

    @property
    def content_start_id(self) -> int:
        return 7

    @property
    def size(self) -> int:
        return int(self.content_start_id + self.base_vocab_size)

    def to_dict(self) -> dict[str, int]:
        return {
            "pad_id": int(self.pad_id),
            "instruction_token_id": int(self.instruction_token_id),
            "input_token_id": int(self.input_token_id),
            "response_token_id": int(self.response_token_id),
            "eos_id": int(self.eos_id),
            "task_token_id": int(self.task_token_id),
            "separator_token_id": int(self.separator_token_id),
            "base_vocab_size": int(self.base_vocab_size),
            "ignore_index": int(self.ignore_index),
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 4096
    batch_size: int = 64
    seq_length: int = 32
    base_vocab_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _next_content_token(token_id: int, vocab: Vocab) -> int:
    base = int(token_id - vocab.content_start_id)
    next_base = (base + 1) % int(vocab.base_vocab_size)
    return int(vocab.content_start_id + next_base)


def _build_example(topic_id: int, vocab: Vocab, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    context_id = _next_content_token(topic_id, vocab)
    response = [
        context_id,
        _next_content_token(context_id, vocab),
        topic_id,
    ]
    tokens = [
        int(vocab.instruction_token_id),
        int(vocab.task_token_id),
        int(topic_id),
        int(vocab.separator_token_id),
        int(vocab.input_token_id),
        int(context_id),
        int(vocab.separator_token_id),
        int(vocab.response_token_id),
        *response,
        int(vocab.eos_id),
    ]

    input_ids = np.full((seq_length,), fill_value=int(vocab.pad_id), dtype=np.int64)
    labels = np.full((seq_length,), fill_value=int(vocab.ignore_index), dtype=np.int64)

    used = min(len(tokens), seq_length)
    input_ids[:used] = np.asarray(tokens[:used], dtype=np.int64)

    response_pos = tokens.index(int(vocab.response_token_id))
    answer_targets = response + [int(vocab.eos_id)]
    for offset, target in enumerate(answer_targets):
        pos = response_pos + offset
        if pos >= seq_length:
            break
        labels[pos] = int(target)

    return input_ids, labels


def _make_dataset_arrays(cfg: DataConfig, vocab: Vocab) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(cfg.seed))
    n = int(cfg.num_samples)
    t = int(cfg.seq_length)

    inputs = np.empty((n, t), dtype=np.int64)
    labels = np.empty((n, t), dtype=np.int64)
    for i in range(n):
        topic_offset = int(rng.integers(0, int(vocab.base_vocab_size)))
        topic_id = int(vocab.content_start_id + topic_offset)
        input_ids, label_ids = _build_example(topic_id=topic_id, vocab=vocab, seq_length=t)
        inputs[i] = input_ids
        labels[i] = label_ids
    return inputs, labels


class ToyInstructionTuningDataset:
    def __init__(self, *, input_ids: np.ndarray, labels: np.ndarray, vocab: Vocab) -> None:
        self.input_ids = input_ids
        self.labels = labels
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int):
        import torch

        input_ids = torch.tensor(self.input_ids[int(idx)], dtype=torch.long)
        labels = torch.tensor(self.labels[int(idx)], dtype=torch.long)
        attention_mask = (input_ids != int(self.vocab.pad_id)).to(torch.float32)
        return {"input_ids": input_ids, "attention_mask": attention_mask}, labels


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    if int(cfg.seq_length) < 12:
        raise ValueError("seq_length must be >= 12 for the toy instruction template")

    vocab = Vocab(base_vocab_size=int(cfg.base_vocab_size))
    input_ids, labels = _make_dataset_arrays(cfg, vocab)
    ds = ToyInstructionTuningDataset(input_ids=input_ids, labels=labels, vocab=vocab)

    train_idx, val_idx = train_val_split_indices(
        n=len(ds), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    def _collate(batch):
        batch_input_ids = torch.stack([b[0]["input_ids"] for b in batch], dim=0)
        batch_attention_mask = torch.stack([b[0]["attention_mask"] for b in batch], dim=0)
        batch_labels = torch.stack([b[1] for b in batch], dim=0)
        return {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask}, batch_labels

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


__all__ = ["DataConfig", "ToyInstructionTuningDataset", "Vocab", "get_dataloaders"]
