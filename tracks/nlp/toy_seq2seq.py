from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class Vocab:
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    base_vocab_size: int = 32

    @property
    def size(self) -> int:
        return 3 + int(self.base_vocab_size)

    def id_to_token(self, idx: int) -> str:
        if idx == self.pad_id:
            return "<pad>"
        if idx == self.bos_id:
            return "<bos>"
        if idx == self.eos_id:
            return "<eos>"
        return f"t{int(idx) - 3}"

    def to_dict(self) -> dict[str, object]:
        return {
            "pad_id": int(self.pad_id),
            "bos_id": int(self.bos_id),
            "eos_id": int(self.eos_id),
            "base_vocab_size": int(self.base_vocab_size),
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 4096
    batch_size: int = 64
    min_len: int = 6
    max_len: int = 18
    base_vocab_size: int = 32
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _make_examples(cfg: DataConfig) -> list[tuple[list[int], list[int]]]:
    rng = np.random.default_rng(int(cfg.seed))

    examples: list[tuple[list[int], list[int]]] = []
    for _ in range(int(cfg.num_samples)):
        length = int(rng.integers(int(cfg.min_len), int(cfg.max_len) + 1))
        src = rng.integers(
            low=3, high=3 + int(cfg.base_vocab_size), size=(length,), dtype=np.int64
        ).tolist()
        tgt = list(reversed(src))
        examples.append((src, tgt))

    rng.shuffle(examples)
    return examples


class ToySeq2SeqDataset:
    def __init__(
        self, *, examples: list[tuple[list[int], list[int]]], vocab: Vocab, cfg: DataConfig
    ) -> None:
        self.examples = list(examples)
        self.vocab = vocab
        self.cfg = cfg
        self.max_src_len = int(cfg.max_len)
        self.max_tgt_len = int(cfg.max_len) + 1  # add <bos>/<eos>

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        import torch

        src, tgt = self.examples[int(idx)]
        pad = int(self.vocab.pad_id)

        src_ids = src[: self.max_src_len]
        src_mask = [1] * len(src_ids)
        while len(src_ids) < self.max_src_len:
            src_ids.append(pad)
            src_mask.append(0)

        tgt_in = [int(self.vocab.bos_id)] + tgt
        tgt_out = tgt + [int(self.vocab.eos_id)]
        tgt_in = tgt_in[: self.max_tgt_len]
        tgt_out = tgt_out[: self.max_tgt_len]

        tgt_mask = [1] * len(tgt_out)
        while len(tgt_in) < self.max_tgt_len:
            tgt_in.append(pad)
        while len(tgt_out) < self.max_tgt_len:
            tgt_out.append(pad)
            tgt_mask.append(0)

        inputs = {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "src_mask": torch.tensor(src_mask, dtype=torch.float32),
            "tgt_in_ids": torch.tensor(tgt_in, dtype=torch.long),
            "tgt_mask": torch.tensor(tgt_mask, dtype=torch.float32),
        }
        targets = {"tgt_out_ids": torch.tensor(tgt_out, dtype=torch.long)}
        return inputs, targets


def get_dataloaders(cfg: DataConfig):
    """Return `(train_loader, val_loader, vocab)` for the toy seq2seq task."""

    import torch
    from torch.utils.data import DataLoader, Subset

    vocab = Vocab(base_vocab_size=int(cfg.base_vocab_size))
    examples = _make_examples(cfg)
    ds = ToySeq2SeqDataset(examples=examples, vocab=vocab, cfg=cfg)

    train_idx, val_idx = train_val_split_indices(
        n=len(ds), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    def _collate(batch):
        src_ids = torch.stack([b[0]["src_ids"] for b in batch], dim=0)
        src_mask = torch.stack([b[0]["src_mask"] for b in batch], dim=0)
        tgt_in_ids = torch.stack([b[0]["tgt_in_ids"] for b in batch], dim=0)
        tgt_mask = torch.stack([b[0]["tgt_mask"] for b in batch], dim=0)
        tgt_out_ids = torch.stack([b[1]["tgt_out_ids"] for b in batch], dim=0)
        return {
            "src_ids": src_ids,
            "src_mask": src_mask,
            "tgt_in_ids": tgt_in_ids,
            "tgt_mask": tgt_mask,
        }, {"tgt_out_ids": tgt_out_ids}

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


__all__ = ["DataConfig", "Vocab", "ToySeq2SeqDataset", "get_dataloaders"]
