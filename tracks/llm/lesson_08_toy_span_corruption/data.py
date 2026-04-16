from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class Vocab:
    pad_id: int = 0
    mask_id: int = 1
    bos_id: int = 2
    target_start_id: int = 3
    eos_id: int = 4
    base_vocab_size: int = 64

    @property
    def content_start_id(self) -> int:
        return 5

    @property
    def size(self) -> int:
        return int(self.content_start_id + self.base_vocab_size)

    def to_dict(self) -> dict[str, int]:
        return {
            "pad_id": int(self.pad_id),
            "mask_id": int(self.mask_id),
            "bos_id": int(self.bos_id),
            "target_start_id": int(self.target_start_id),
            "eos_id": int(self.eos_id),
            "base_vocab_size": int(self.base_vocab_size),
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 4096
    batch_size: int = 64
    seq_length: int = 24
    base_vocab_size: int = 64
    mask_ratio: float = 0.25
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _clip_span_len(*, source_len: int, mask_ratio: float) -> int:
    raw = int(round(float(mask_ratio) * float(source_len)))
    return int(max(1, min(source_len - 1, raw)))


def _build_example(*, seq_length: int, vocab: Vocab, mask_ratio: float, rng) -> tuple[np.ndarray, np.ndarray]:
    t = int(seq_length)
    source_len = int(t - 4)
    source = rng.integers(
        low=int(vocab.content_start_id),
        high=int(vocab.size),
        size=(source_len,),
        dtype=np.int64,
    )

    span_len = _clip_span_len(source_len=source_len, mask_ratio=mask_ratio)
    span_start = int(rng.integers(0, source_len - span_len + 1))
    span_end = int(span_start + span_len)

    target_span = source[span_start:span_end]
    corrupted = np.concatenate(
        [
            source[:span_start],
            np.asarray([int(vocab.mask_id)], dtype=np.int64),
            source[span_end:],
        ],
        axis=0,
    )

    prompt = np.concatenate(
        [
            np.asarray([int(vocab.bos_id)], dtype=np.int64),
            corrupted,
            np.asarray([int(vocab.target_start_id)], dtype=np.int64),
        ],
        axis=0,
    )
    target = np.concatenate(
        [
            target_span,
            np.asarray([int(vocab.eos_id)], dtype=np.int64),
        ],
        axis=0,
    )

    input_ids = np.full((t,), fill_value=int(vocab.pad_id), dtype=np.int64)
    labels = np.full((t,), fill_value=-100, dtype=np.int64)

    merged = np.concatenate([prompt, target], axis=0)[:t]
    input_ids[: merged.shape[0]] = merged

    target_start = min(prompt.shape[0], t)
    target_tokens = merged[target_start:]
    labels[target_start : target_start + target_tokens.shape[0]] = target_tokens
    return input_ids, labels


def _make_dataset_arrays(cfg: DataConfig, vocab: Vocab) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(cfg.seed))
    n = int(cfg.num_samples)
    t = int(cfg.seq_length)

    input_ids = np.empty((n, t), dtype=np.int64)
    attention_mask = np.ones((n, t), dtype=np.float32)
    labels = np.empty((n, t), dtype=np.int64)

    for idx in range(n):
        ids, target = _build_example(
            seq_length=t,
            vocab=vocab,
            mask_ratio=float(cfg.mask_ratio),
            rng=rng,
        )
        input_ids[idx] = ids
        labels[idx] = target
    return input_ids, attention_mask, labels


class ToySpanCorruptionDataset:
    def __init__(
        self,
        *,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int):
        import torch

        i = int(idx)
        return {
            "input_ids": torch.tensor(self.input_ids[i], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[i], dtype=torch.float32),
            "labels": torch.tensor(self.labels[i], dtype=torch.long),
        }


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    if int(cfg.seq_length) < 8:
        raise ValueError("seq_length must be >= 8 for toy span-corruption sequences")
    if int(cfg.base_vocab_size) < 8:
        raise ValueError("base_vocab_size must be >= 8")

    vocab = Vocab(base_vocab_size=int(cfg.base_vocab_size))
    input_ids, attention_mask, labels = _make_dataset_arrays(cfg, vocab)
    ds = ToySpanCorruptionDataset(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    train_idx, val_idx = train_val_split_indices(
        n=len(ds), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    def _collate(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
            "labels": torch.stack([b["labels"] for b in batch], dim=0),
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


__all__ = ["DataConfig", "ToySpanCorruptionDataset", "Vocab", "get_dataloaders"]
