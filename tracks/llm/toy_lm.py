from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class Vocab:
    pad_id: int = 0
    base_vocab_size: int = 64

    @property
    def size(self) -> int:
        return 1 + int(self.base_vocab_size)

    def to_dict(self) -> dict[str, object]:
        return {"pad_id": int(self.pad_id), "base_vocab_size": int(self.base_vocab_size)}


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 4096
    batch_size: int = 64
    seq_length: int = 64  # includes a final padding token for shifting
    base_vocab_size: int = 64
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _make_sequences(cfg: DataConfig) -> np.ndarray:
    """Generate sequences that follow a simple rule: token increments by 1 (mod V)."""

    rng = np.random.default_rng(int(cfg.seed))
    n = int(cfg.num_samples)
    t = int(cfg.seq_length)
    v = int(cfg.base_vocab_size)

    # We generate (t-1) valid tokens, then one padding token.
    out = np.empty((n, t), dtype=np.int64)
    for i in range(n):
        start = int(rng.integers(1, 1 + v))  # token ids in [1, V]
        seq = (start + np.arange(t - 1)) % v + 1  # stay in [1, V]
        out[i, : t - 1] = seq
        out[i, t - 1] = 0  # pad
    return out


class ToyLMDataset:
    def __init__(self, *, sequences: np.ndarray, vocab: Vocab) -> None:
        self.sequences = sequences
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.sequences.shape[0])

    def __getitem__(self, idx: int):
        import torch

        seq = self.sequences[int(idx)]
        input_ids = torch.tensor(seq, dtype=torch.long)
        attention_mask = (input_ids != int(self.vocab.pad_id)).to(torch.float32)

        # Next-token prediction: labels are shifted left, last is pad (ignored).
        labels = torch.empty_like(input_ids)
        labels[:-1] = input_ids[1:]
        labels[-1] = int(self.vocab.pad_id)

        return {"input_ids": input_ids, "attention_mask": attention_mask}, labels


def get_dataloaders(cfg: DataConfig):
    """Return `(train_loader, val_loader, vocab)` for the toy LM task."""

    import torch
    from torch.utils.data import DataLoader, Subset

    if int(cfg.seq_length) < 4:
        raise ValueError("seq_length must be >= 4")

    vocab = Vocab(base_vocab_size=int(cfg.base_vocab_size))
    sequences = _make_sequences(cfg)
    ds = ToyLMDataset(sequences=sequences, vocab=vocab)

    train_idx, val_idx = train_val_split_indices(
        n=len(ds), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    def _collate(batch):
        input_ids = torch.stack([b[0]["input_ids"] for b in batch], dim=0)
        attention_mask = torch.stack([b[0]["attention_mask"] for b in batch], dim=0)
        labels = torch.stack([b[1] for b in batch], dim=0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}, labels

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


__all__ = ["DataConfig", "Vocab", "ToyLMDataset", "get_dataloaders"]
