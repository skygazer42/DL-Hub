from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class Vocab:
    pad_id: int = 0
    prompt_token_id: int = 1
    chosen_token_id: int = 2
    rejected_token_id: int = 3
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
            "prompt_token_id": int(self.prompt_token_id),
            "chosen_token_id": int(self.chosen_token_id),
            "rejected_token_id": int(self.rejected_token_id),
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


def _build_pair(topic_id: int, vocab: Vocab, seq_length: int) -> tuple[np.ndarray, ...]:
    c1 = _next_content_token(topic_id, vocab)
    c2 = _next_content_token(c1, vocab)
    chosen_response = [int(topic_id), int(c1), int(c2)]
    rejected_response = [int(c1), int(topic_id), int(c2)]

    prompt = [
        int(vocab.prompt_token_id),
        int(vocab.task_token_id),
        int(topic_id),
        int(vocab.separator_token_id),
    ]
    chosen_tokens = [*prompt, int(vocab.chosen_token_id), *chosen_response, int(vocab.eos_id)]
    rejected_tokens = [*prompt, int(vocab.rejected_token_id), *rejected_response, int(vocab.eos_id)]

    chosen_input_ids = np.full((seq_length,), fill_value=int(vocab.pad_id), dtype=np.int64)
    rejected_input_ids = np.full((seq_length,), fill_value=int(vocab.pad_id), dtype=np.int64)
    chosen_labels = np.full((seq_length,), fill_value=int(vocab.ignore_index), dtype=np.int64)
    rejected_labels = np.full((seq_length,), fill_value=int(vocab.ignore_index), dtype=np.int64)

    used_chosen = min(len(chosen_tokens), seq_length)
    used_rejected = min(len(rejected_tokens), seq_length)
    chosen_input_ids[:used_chosen] = np.asarray(chosen_tokens[:used_chosen], dtype=np.int64)
    rejected_input_ids[:used_rejected] = np.asarray(rejected_tokens[:used_rejected], dtype=np.int64)

    chosen_start = chosen_tokens.index(int(vocab.chosen_token_id))
    rejected_start = rejected_tokens.index(int(vocab.rejected_token_id))
    chosen_targets = chosen_response + [int(vocab.eos_id)]
    rejected_targets = rejected_response + [int(vocab.eos_id)]
    for offset, target in enumerate(chosen_targets):
        pos = chosen_start + offset
        if pos >= seq_length:
            break
        chosen_labels[pos] = int(target)
    for offset, target in enumerate(rejected_targets):
        pos = rejected_start + offset
        if pos >= seq_length:
            break
        rejected_labels[pos] = int(target)

    return chosen_input_ids, rejected_input_ids, chosen_labels, rejected_labels


def _make_dataset_arrays(cfg: DataConfig, vocab: Vocab) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(int(cfg.seed))
    n = int(cfg.num_samples)
    t = int(cfg.seq_length)

    chosen_inputs = np.empty((n, t), dtype=np.int64)
    rejected_inputs = np.empty((n, t), dtype=np.int64)
    chosen_labels = np.empty((n, t), dtype=np.int64)
    rejected_labels = np.empty((n, t), dtype=np.int64)
    for i in range(n):
        topic_offset = int(rng.integers(0, int(vocab.base_vocab_size)))
        topic_id = int(vocab.content_start_id + topic_offset)
        c_inp, r_inp, c_lbl, r_lbl = _build_pair(topic_id=topic_id, vocab=vocab, seq_length=t)
        chosen_inputs[i] = c_inp
        rejected_inputs[i] = r_inp
        chosen_labels[i] = c_lbl
        rejected_labels[i] = r_lbl

    return chosen_inputs, rejected_inputs, chosen_labels, rejected_labels


class SyntheticPreferenceDataset:
    def __init__(
        self,
        *,
        chosen_input_ids: np.ndarray,
        rejected_input_ids: np.ndarray,
        chosen_labels: np.ndarray,
        rejected_labels: np.ndarray,
        vocab: Vocab,
    ) -> None:
        self.chosen_input_ids = chosen_input_ids
        self.rejected_input_ids = rejected_input_ids
        self.chosen_labels = chosen_labels
        self.rejected_labels = rejected_labels
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.chosen_input_ids.shape[0])

    def __getitem__(self, idx: int):
        import torch

        chosen_input_ids = torch.tensor(self.chosen_input_ids[int(idx)], dtype=torch.long)
        rejected_input_ids = torch.tensor(self.rejected_input_ids[int(idx)], dtype=torch.long)
        chosen_labels = torch.tensor(self.chosen_labels[int(idx)], dtype=torch.long)
        rejected_labels = torch.tensor(self.rejected_labels[int(idx)], dtype=torch.long)

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": (chosen_input_ids != int(self.vocab.pad_id)).to(torch.float32),
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": (rejected_input_ids != int(self.vocab.pad_id)).to(torch.float32),
            "rejected_labels": rejected_labels,
        }


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    if int(cfg.seq_length) < 10:
        raise ValueError("seq_length must be >= 10 for the synthetic preference template")

    vocab = Vocab(base_vocab_size=int(cfg.base_vocab_size))
    arrays = _make_dataset_arrays(cfg, vocab)
    ds = SyntheticPreferenceDataset(
        chosen_input_ids=arrays[0],
        rejected_input_ids=arrays[1],
        chosen_labels=arrays[2],
        rejected_labels=arrays[3],
        vocab=vocab,
    )

    train_idx, val_idx = train_val_split_indices(
        n=len(ds), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    def _collate(batch):
        out: dict[str, torch.Tensor] = {}
        keys = batch[0].keys()
        for key in keys:
            out[key] = torch.stack([row[key] for row in batch], dim=0)
        return out

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
