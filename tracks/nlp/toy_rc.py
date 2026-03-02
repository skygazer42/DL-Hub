from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_id: int
    unk_id: int

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def encode_tokens(self, tokens: list[str], *, max_length: int) -> tuple[list[int], list[int]]:
        ids = [self.token_to_id.get(t, self.unk_id) for t in tokens][: int(max_length)]
        attn = [1] * len(ids)
        while len(ids) < int(max_length):
            ids.append(self.pad_id)
            attn.append(0)
        return ids, attn

    def to_dict(self) -> dict[str, object]:
        return {"pad_id": self.pad_id, "unk_id": self.unk_id, "token_to_id": dict(self.token_to_id)}


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 2048
    batch_size: int = 64
    context_length: int = 32
    question_length: int = 4
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _make_examples(cfg: DataConfig) -> list[tuple[list[str], list[str], int, int]]:
    rng = np.random.default_rng(int(cfg.seed))
    n = int(cfg.num_samples)
    l_ctx = int(cfg.context_length)

    keys = [f"key{i}" for i in range(24)]
    fillers = [f"w{i}" for i in range(80)]
    modes = ["first", "last"]

    examples: list[tuple[list[str], list[str], int, int]] = []
    for _ in range(n):
        key = str(rng.choice(keys))
        mode = str(rng.choice(modes))

        ctx = rng.choice(fillers, size=l_ctx, replace=True).tolist()

        # Insert the key multiple times. The model must select either the first or last occurrence.
        num_occurs = int(rng.integers(low=2, high=5))
        positions = rng.choice(np.arange(l_ctx), size=num_occurs, replace=False)
        positions = sorted(int(p) for p in positions)
        for p in positions:
            ctx[p] = key

        start = positions[0] if mode == "first" else positions[-1]
        end = start
        q = [mode, key]
        examples.append((ctx, q, int(start), int(end)))

    rng.shuffle(examples)
    return examples


def _build_vocab(examples: list[tuple[list[str], list[str], int, int]]) -> Vocab:
    tokens: set[str] = set()
    for ctx, q, _, _ in examples:
        tokens.update(ctx)
        tokens.update(q)

    id_to_token = ["<pad>", "<unk>"]
    token_to_id = {"<pad>": 0, "<unk>": 1}
    for tok in sorted(tokens):
        if tok in token_to_id:
            continue
        token_to_id[tok] = len(id_to_token)
        id_to_token.append(tok)
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token, pad_id=0, unk_id=1)


class ToyRCDataset:
    def __init__(self, *, examples: list[tuple[list[str], list[str], int, int]], vocab: Vocab, cfg: DataConfig) -> None:
        self.examples = list(examples)
        self.vocab = vocab
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        import torch

        ctx, q, start, end = self.examples[int(idx)]
        ctx_ids, ctx_mask = self.vocab.encode_tokens(ctx, max_length=int(self.cfg.context_length))
        q_ids, q_mask = self.vocab.encode_tokens(q, max_length=int(self.cfg.question_length))

        inputs = {
            "context_ids": torch.tensor(ctx_ids, dtype=torch.long),
            "context_mask": torch.tensor(ctx_mask, dtype=torch.float32),
            "question_ids": torch.tensor(q_ids, dtype=torch.long),
            "question_mask": torch.tensor(q_mask, dtype=torch.float32),
        }
        targets = {
            "start": torch.tensor(int(start), dtype=torch.long),
            "end": torch.tensor(int(end), dtype=torch.long),
        }
        return inputs, targets


def get_dataloaders(cfg: DataConfig):
    """Return `(train_loader, val_loader, vocab)` for the toy reading comprehension task."""

    import torch
    from torch.utils.data import DataLoader, Subset

    examples = _make_examples(cfg)
    vocab = _build_vocab(examples)
    ds = ToyRCDataset(examples=examples, vocab=vocab, cfg=cfg)

    train_idx, val_idx = train_val_split_indices(n=len(ds), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed))
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    def _collate(batch):
        context_ids = torch.stack([b[0]["context_ids"] for b in batch], dim=0)
        context_mask = torch.stack([b[0]["context_mask"] for b in batch], dim=0)
        question_ids = torch.stack([b[0]["question_ids"] for b in batch], dim=0)
        question_mask = torch.stack([b[0]["question_mask"] for b in batch], dim=0)

        start = torch.stack([b[1]["start"] for b in batch], dim=0)
        end = torch.stack([b[1]["end"] for b in batch], dim=0)

        return (
            {
                "context_ids": context_ids,
                "context_mask": context_mask,
                "question_ids": question_ids,
                "question_mask": question_mask,
            },
            {"start": start, "end": end},
        )

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


__all__ = ["DataConfig", "Vocab", "ToyRCDataset", "get_dataloaders"]

