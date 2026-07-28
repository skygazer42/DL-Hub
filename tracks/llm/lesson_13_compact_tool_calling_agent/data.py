from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class Vocab:
    pad_id: int = 0
    user_token_id: int = 1
    call_token_id: int = 2
    calc_tool_token_id: int = 3
    lookup_tool_token_id: int = 4
    result_token_id: int = 5
    eos_id: int = 6
    base_vocab_size: int = 64
    num_tools: int = 2
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
            "user_token_id": int(self.user_token_id),
            "call_token_id": int(self.call_token_id),
            "calc_tool_token_id": int(self.calc_tool_token_id),
            "lookup_tool_token_id": int(self.lookup_tool_token_id),
            "result_token_id": int(self.result_token_id),
            "eos_id": int(self.eos_id),
            "base_vocab_size": int(self.base_vocab_size),
            "num_tools": int(self.num_tools),
            "ignore_index": int(self.ignore_index),
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 4096
    batch_size: int = 64
    seq_length: int = 24
    base_vocab_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _encode_number(value: int, vocab: Vocab) -> int:
    base = int(value) % int(vocab.base_vocab_size)
    return int(vocab.content_start_id + base)


def _lookup_value(a: int, b: int, vocab: Vocab) -> int:
    key = ((int(a) * 10) + int(b)) % int(vocab.base_vocab_size)
    return (3 * key + 1) % int(vocab.base_vocab_size)


def _execute_tool(tool_id: int, a: int, b: int, vocab: Vocab) -> int:
    if int(tool_id) == 0:
        return (int(a) + int(b)) % int(vocab.base_vocab_size)
    return _lookup_value(a, b, vocab)


def _build_sample(*, tool_id: int, a: int, b: int, vocab: Vocab, seq_length: int) -> tuple[np.ndarray, ...]:
    tool_token = int(vocab.calc_tool_token_id) if int(tool_id) == 0 else int(vocab.lookup_tool_token_id)
    result_value = _execute_tool(tool_id=tool_id, a=a, b=b, vocab=vocab)
    result_token = _encode_number(result_value, vocab)

    tokens = [
        int(vocab.user_token_id),
        _encode_number(a, vocab),
        _encode_number(b, vocab),
        int(vocab.call_token_id),
        tool_token,
        int(vocab.result_token_id),
        result_token,
        int(vocab.eos_id),
    ]
    input_ids = np.full((seq_length,), fill_value=int(vocab.pad_id), dtype=np.int64)
    labels = np.full((seq_length,), fill_value=int(vocab.ignore_index), dtype=np.int64)

    used = min(int(seq_length), len(tokens))
    input_ids[:used] = np.asarray(tokens[:used], dtype=np.int64)
    if used > 1:
        labels[: (used - 1)] = input_ids[1:used]
    return input_ids, labels, np.asarray(tool_id, dtype=np.int64)


def _make_dataset_arrays(cfg: DataConfig, vocab: Vocab) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(int(cfg.seed))
    n = int(cfg.num_samples)
    t = int(cfg.seq_length)

    input_ids = np.empty((n, t), dtype=np.int64)
    labels = np.empty((n, t), dtype=np.int64)
    tool_targets = np.empty((n,), dtype=np.int64)

    for idx in range(n):
        a = int(rng.integers(0, int(vocab.base_vocab_size)))
        b = int(rng.integers(0, int(vocab.base_vocab_size)))
        tool_id = int(rng.integers(0, int(vocab.num_tools)))
        ids_i, labels_i, tool_i = _build_sample(
            tool_id=tool_id,
            a=a,
            b=b,
            vocab=vocab,
            seq_length=t,
        )
        input_ids[idx] = ids_i
        labels[idx] = labels_i
        tool_targets[idx] = tool_i
    return input_ids, labels, tool_targets


class SyntheticToolCallingDataset:
    def __init__(
        self,
        *,
        input_ids: np.ndarray,
        labels: np.ndarray,
        tool_targets: np.ndarray,
        vocab: Vocab,
    ) -> None:
        self.input_ids = input_ids
        self.labels = labels
        self.tool_targets = tool_targets
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int):
        import torch

        ids = torch.tensor(self.input_ids[int(idx)], dtype=torch.long)
        labels = torch.tensor(self.labels[int(idx)], dtype=torch.long)
        tool_target = torch.tensor(self.tool_targets[int(idx)], dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": (ids != int(self.vocab.pad_id)).to(torch.float32),
            "labels": labels,
            "tool_targets": tool_target,
        }


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    if int(cfg.seq_length) < 8:
        raise ValueError("seq_length must be >= 8 for synthetic tool-calling sequences")

    vocab = Vocab(base_vocab_size=int(cfg.base_vocab_size))
    input_ids, labels, tool_targets = _make_dataset_arrays(cfg, vocab)
    ds = SyntheticToolCallingDataset(
        input_ids=input_ids,
        labels=labels,
        tool_targets=tool_targets,
        vocab=vocab,
    )
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
            "tool_targets": torch.stack([row["tool_targets"] for row in batch], dim=0),
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


__all__ = ["DataConfig", "SyntheticToolCallingDataset", "Vocab", "get_dataloaders"]
