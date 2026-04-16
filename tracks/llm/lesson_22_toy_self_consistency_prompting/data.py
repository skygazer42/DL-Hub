from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class Vocab:
    pad_id: int = 0
    prompt_token_id: int = 1
    sample_token_id: int = 2
    agree_token_id: int = 3
    disagree_token_id: int = 4
    vote_token_id: int = 5
    separator_token_id: int = 6
    eos_id: int = 7
    base_vocab_size: int = 64
    ignore_index: int = -100

    @property
    def content_start_id(self) -> int:
        return 8

    @property
    def size(self) -> int:
        return int(self.content_start_id + self.base_vocab_size)

    def to_dict(self) -> dict[str, int]:
        return {
            "pad_id": int(self.pad_id),
            "prompt_token_id": int(self.prompt_token_id),
            "sample_token_id": int(self.sample_token_id),
            "agree_token_id": int(self.agree_token_id),
            "disagree_token_id": int(self.disagree_token_id),
            "vote_token_id": int(self.vote_token_id),
            "separator_token_id": int(self.separator_token_id),
            "eos_id": int(self.eos_id),
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


def _offset_content_token(token_id: int, *, offset: int, vocab: Vocab) -> int:
    base = int(token_id - vocab.content_start_id)
    return int(vocab.content_start_id + ((base + int(offset)) % int(vocab.base_vocab_size)))


def _build_example(*, prompt_id: int, vocab: Vocab, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    candidate_majority = [
        _offset_content_token(prompt_id, offset=1, vocab=vocab),
        _offset_content_token(prompt_id, offset=2, vocab=vocab),
    ]
    candidate_outlier = [
        _offset_content_token(prompt_id, offset=3, vocab=vocab),
        _offset_content_token(prompt_id, offset=4, vocab=vocab),
    ]
    final_answer = [*candidate_majority, _offset_content_token(prompt_id, offset=5, vocab=vocab)]

    tokens = [
        int(vocab.prompt_token_id),
        int(prompt_id),
        int(vocab.separator_token_id),
        int(vocab.sample_token_id),
        *candidate_majority,
        int(vocab.agree_token_id),
        int(vocab.separator_token_id),
        int(vocab.sample_token_id),
        *candidate_majority,
        int(vocab.agree_token_id),
        int(vocab.separator_token_id),
        int(vocab.sample_token_id),
        *candidate_outlier,
        int(vocab.disagree_token_id),
        int(vocab.separator_token_id),
        int(vocab.vote_token_id),
        *final_answer,
        int(vocab.eos_id),
    ]

    input_ids = np.full((int(seq_length),), fill_value=int(vocab.pad_id), dtype=np.int64)
    labels = np.full((int(seq_length),), fill_value=int(vocab.ignore_index), dtype=np.int64)
    used = min(len(tokens), int(seq_length))
    input_ids[:used] = np.asarray(tokens[:used], dtype=np.int64)

    vote_pos = tokens.index(int(vocab.vote_token_id))
    targets = final_answer + [int(vocab.eos_id)]
    for offset, token_id in enumerate(targets):
        pos = vote_pos + offset
        if pos >= int(seq_length):
            break
        labels[pos] = int(token_id)
    return input_ids, labels


def _make_dataset_arrays(cfg: DataConfig, vocab: Vocab) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(cfg.seed))
    n = int(cfg.num_samples)
    t = int(cfg.seq_length)

    input_ids = np.empty((n, t), dtype=np.int64)
    labels = np.empty((n, t), dtype=np.int64)
    for idx in range(n):
        prompt_offset = int(rng.integers(0, int(vocab.base_vocab_size)))
        prompt_id = int(vocab.content_start_id + prompt_offset)
        ids, target = _build_example(prompt_id=prompt_id, vocab=vocab, seq_length=t)
        input_ids[idx] = ids
        labels[idx] = target
    return input_ids, labels


class ToySelfConsistencyPromptingDataset:
    def __init__(self, *, input_ids: np.ndarray, labels: np.ndarray, vocab: Vocab) -> None:
        self.input_ids = input_ids
        self.labels = labels
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int):
        import torch

        ids = torch.tensor(self.input_ids[int(idx)], dtype=torch.long)
        labels = torch.tensor(self.labels[int(idx)], dtype=torch.long)
        attention_mask = (ids != int(self.vocab.pad_id)).to(torch.float32)
        return {"input_ids": ids, "attention_mask": attention_mask}, labels


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    if int(cfg.seq_length) < 22:
        raise ValueError("seq_length must be >= 22 for toy self-consistency prompting")

    vocab = Vocab(base_vocab_size=int(cfg.base_vocab_size))
    input_ids, labels = _make_dataset_arrays(cfg, vocab)
    dataset = ToySelfConsistencyPromptingDataset(input_ids=input_ids, labels=labels, vocab=vocab)

    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )

    def _collate(batch):
        batch_input_ids = torch.stack([row[0]["input_ids"] for row in batch], dim=0)
        batch_attention_mask = torch.stack([row[0]["attention_mask"] for row in batch], dim=0)
        batch_labels = torch.stack([row[1] for row in batch], dim=0)
        return {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask}, batch_labels

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


__all__ = ["DataConfig", "ToySelfConsistencyPromptingDataset", "Vocab", "get_dataloaders"]
