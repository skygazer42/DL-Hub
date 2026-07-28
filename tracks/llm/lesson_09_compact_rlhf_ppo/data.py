from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class Vocab:
    pad_id: int = 0
    prompt_token_id: int = 1
    good_token_id: int = 2
    bad_token_id: int = 3
    separator_token_id: int = 4
    eos_id: int = 5
    base_vocab_size: int = 64
    ignore_index: int = -100

    @property
    def content_start_id(self) -> int:
        return 6

    @property
    def size(self) -> int:
        return int(self.content_start_id + self.base_vocab_size)

    def to_dict(self) -> dict[str, int]:
        return {
            "pad_id": int(self.pad_id),
            "prompt_token_id": int(self.prompt_token_id),
            "good_token_id": int(self.good_token_id),
            "bad_token_id": int(self.bad_token_id),
            "separator_token_id": int(self.separator_token_id),
            "eos_id": int(self.eos_id),
            "base_vocab_size": int(self.base_vocab_size),
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


def _next_content_token(token_id: int, vocab: Vocab) -> int:
    base = int(token_id - vocab.content_start_id)
    next_base = (base + 1) % int(vocab.base_vocab_size)
    return int(vocab.content_start_id + next_base)


def _prev_content_token(token_id: int, vocab: Vocab) -> int:
    base = int(token_id - vocab.content_start_id)
    prev_base = (base - 1) % int(vocab.base_vocab_size)
    return int(vocab.content_start_id + prev_base)


def _build_sample(topic_id: int, use_good_response: bool, vocab: Vocab, seq_length: int) -> tuple[np.ndarray, ...]:
    prompt = [int(vocab.prompt_token_id), int(topic_id), int(vocab.separator_token_id)]
    if use_good_response:
        response_tokens = [
            int(vocab.good_token_id),
            int(topic_id),
            _next_content_token(topic_id, vocab),
            _next_content_token(_next_content_token(topic_id, vocab), vocab),
            int(vocab.eos_id),
        ]
    else:
        response_tokens = [
            int(vocab.bad_token_id),
            int(topic_id),
            _prev_content_token(topic_id, vocab),
            _prev_content_token(_prev_content_token(topic_id, vocab), vocab),
            int(vocab.eos_id),
        ]

    tokens = prompt + response_tokens
    input_ids = np.full((seq_length,), fill_value=int(vocab.pad_id), dtype=np.int64)
    labels = np.full((seq_length,), fill_value=int(vocab.ignore_index), dtype=np.int64)
    response_mask = np.zeros((seq_length,), dtype=np.float32)

    used = min(len(tokens), seq_length)
    input_ids[:used] = np.asarray(tokens[:used], dtype=np.int64)

    # Only optimize over response content tokens using PPO-style token log-prob ratios.
    targets = response_tokens[1:]
    start = len(prompt)
    for offset, target in enumerate(targets):
        pos = start + offset
        if pos >= seq_length:
            break
        labels[pos] = int(target)
        response_mask[pos] = 1.0
    return input_ids, labels, response_mask


def _make_dataset_arrays(cfg: DataConfig, vocab: Vocab) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(int(cfg.seed))
    n = int(cfg.num_samples)
    t = int(cfg.seq_length)

    input_ids = np.empty((n, t), dtype=np.int64)
    labels = np.empty((n, t), dtype=np.int64)
    response_mask = np.empty((n, t), dtype=np.float32)

    for idx in range(n):
        topic_offset = int(rng.integers(0, int(vocab.base_vocab_size)))
        topic_id = int(vocab.content_start_id + topic_offset)
        use_good_response = bool(rng.integers(0, 2))
        ids_i, labels_i, mask_i = _build_sample(
            topic_id=topic_id,
            use_good_response=use_good_response,
            vocab=vocab,
            seq_length=t,
        )
        input_ids[idx] = ids_i
        labels[idx] = labels_i
        response_mask[idx] = mask_i
    return input_ids, labels, response_mask


class SyntheticRlhfPpoDataset:
    def __init__(
        self,
        *,
        input_ids: np.ndarray,
        labels: np.ndarray,
        response_mask: np.ndarray,
        vocab: Vocab,
    ) -> None:
        self.input_ids = input_ids
        self.labels = labels
        self.response_mask = response_mask
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int):
        import torch

        ids = torch.tensor(self.input_ids[int(idx)], dtype=torch.long)
        labels = torch.tensor(self.labels[int(idx)], dtype=torch.long)
        response_mask = torch.tensor(self.response_mask[int(idx)], dtype=torch.float32)
        return {
            "input_ids": ids,
            "attention_mask": (ids != int(self.vocab.pad_id)).to(torch.float32),
            "labels": labels,
            "response_mask": response_mask,
        }


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    if int(cfg.seq_length) < 8:
        raise ValueError("seq_length must be >= 8 for synthetic PPO RLHF sequences")

    vocab = Vocab(base_vocab_size=int(cfg.base_vocab_size))
    input_ids, labels, response_mask = _make_dataset_arrays(cfg, vocab)
    ds = SyntheticRlhfPpoDataset(
        input_ids=input_ids,
        labels=labels,
        response_mask=response_mask,
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
            "response_mask": torch.stack([row["response_mask"] for row in batch], dim=0),
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


__all__ = ["DataConfig", "SyntheticRlhfPpoDataset", "Vocab", "get_dataloaders"]
