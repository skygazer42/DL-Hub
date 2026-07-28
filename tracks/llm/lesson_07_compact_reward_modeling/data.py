from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class Vocab:
    pad_id: int = 0
    prompt_token_id: int = 1
    chosen_token_id: int = 2
    rejected_token_id: int = 3
    separator_token_id: int = 4
    eos_id: int = 5
    base_vocab_size: int = 64

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
            "chosen_token_id": int(self.chosen_token_id),
            "rejected_token_id": int(self.rejected_token_id),
            "separator_token_id": int(self.separator_token_id),
            "eos_id": int(self.eos_id),
            "base_vocab_size": int(self.base_vocab_size),
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


def _build_pair(topic_id: int, vocab: Vocab, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    prompt = [int(vocab.prompt_token_id), int(topic_id), int(vocab.separator_token_id)]
    chosen_completion = [
        int(vocab.chosen_token_id),
        int(topic_id),
        _next_content_token(topic_id, vocab),
        _next_content_token(_next_content_token(topic_id, vocab), vocab),
        int(vocab.eos_id),
    ]
    rejected_completion = [
        int(vocab.rejected_token_id),
        int(topic_id),
        _prev_content_token(topic_id, vocab),
        _prev_content_token(_prev_content_token(topic_id, vocab), vocab),
        int(vocab.eos_id),
    ]

    chosen_tokens = prompt + chosen_completion
    rejected_tokens = prompt + rejected_completion

    chosen_ids = np.full((seq_length,), fill_value=int(vocab.pad_id), dtype=np.int64)
    rejected_ids = np.full((seq_length,), fill_value=int(vocab.pad_id), dtype=np.int64)

    chosen_used = min(len(chosen_tokens), seq_length)
    rejected_used = min(len(rejected_tokens), seq_length)
    chosen_ids[:chosen_used] = np.asarray(chosen_tokens[:chosen_used], dtype=np.int64)
    rejected_ids[:rejected_used] = np.asarray(rejected_tokens[:rejected_used], dtype=np.int64)
    return chosen_ids, rejected_ids


def _make_dataset_arrays(cfg: DataConfig, vocab: Vocab) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(cfg.seed))
    n = int(cfg.num_samples)
    t = int(cfg.seq_length)

    chosen = np.empty((n, t), dtype=np.int64)
    rejected = np.empty((n, t), dtype=np.int64)
    for idx in range(n):
        topic_offset = int(rng.integers(0, int(vocab.base_vocab_size)))
        topic_id = int(vocab.content_start_id + topic_offset)
        chosen_ids, rejected_ids = _build_pair(topic_id=topic_id, vocab=vocab, seq_length=t)
        chosen[idx] = chosen_ids
        rejected[idx] = rejected_ids
    return chosen, rejected


class SyntheticPreferenceDataset:
    def __init__(self, *, chosen_input_ids: np.ndarray, rejected_input_ids: np.ndarray, vocab: Vocab) -> None:
        self.chosen_input_ids = chosen_input_ids
        self.rejected_input_ids = rejected_input_ids
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.chosen_input_ids.shape[0])

    def __getitem__(self, idx: int):
        import torch

        chosen_ids = torch.tensor(self.chosen_input_ids[int(idx)], dtype=torch.long)
        rejected_ids = torch.tensor(self.rejected_input_ids[int(idx)], dtype=torch.long)
        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": (chosen_ids != int(self.vocab.pad_id)).to(torch.float32),
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": (rejected_ids != int(self.vocab.pad_id)).to(torch.float32),
        }


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    if int(cfg.seq_length) < 8:
        raise ValueError("seq_length must be >= 8 for synthetic preference pairs")

    vocab = Vocab(base_vocab_size=int(cfg.base_vocab_size))
    chosen_ids, rejected_ids = _make_dataset_arrays(cfg, vocab)
    ds = SyntheticPreferenceDataset(
        chosen_input_ids=chosen_ids,
        rejected_input_ids=rejected_ids,
        vocab=vocab,
    )

    train_idx, val_idx = train_val_split_indices(
        n=len(ds), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    def _collate(batch):
        return {
            "chosen_input_ids": torch.stack([b["chosen_input_ids"] for b in batch], dim=0),
            "chosen_attention_mask": torch.stack([b["chosen_attention_mask"] for b in batch], dim=0),
            "rejected_input_ids": torch.stack([b["rejected_input_ids"] for b in batch], dim=0),
            "rejected_attention_mask": torch.stack([b["rejected_attention_mask"] for b in batch], dim=0),
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


__all__ = ["DataConfig", "SyntheticPreferenceDataset", "Vocab", "get_dataloaders"]
