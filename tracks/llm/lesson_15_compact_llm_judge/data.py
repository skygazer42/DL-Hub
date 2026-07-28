from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices


@dataclass(frozen=True)
class Vocab:
    pad_id: int = 0
    prompt_token_id: int = 1
    ref_token_id: int = 2
    candidate_token_id: int = 3
    verdict_token_id: int = 4
    good_token_id: int = 5
    bad_token_id: int = 6
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
            "ref_token_id": int(self.ref_token_id),
            "candidate_token_id": int(self.candidate_token_id),
            "verdict_token_id": int(self.verdict_token_id),
            "good_token_id": int(self.good_token_id),
            "bad_token_id": int(self.bad_token_id),
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


def _encode_content(value: int, vocab: Vocab) -> int:
    base = int(value) % int(vocab.base_vocab_size)
    return int(vocab.content_start_id + base)


def _judge_score(reference_value: int, candidate_value: int, vocab: Vocab) -> float:
    ref = int(reference_value) % int(vocab.base_vocab_size)
    cand = int(candidate_value) % int(vocab.base_vocab_size)
    distance = min(abs(ref - cand), int(vocab.base_vocab_size) - abs(ref - cand))
    return 1.0 if distance <= 2 else 0.0


def _build_sample(*, prompt_value: int, reference_value: int, candidate_value: int, vocab: Vocab, seq_length: int):
    score = _judge_score(reference_value, candidate_value, vocab)
    verdict_token = int(vocab.good_token_id) if score >= 0.5 else int(vocab.bad_token_id)
    tokens = [
        int(vocab.prompt_token_id),
        _encode_content(prompt_value, vocab),
        int(vocab.ref_token_id),
        _encode_content(reference_value, vocab),
        int(vocab.candidate_token_id),
        _encode_content(candidate_value, vocab),
        int(vocab.verdict_token_id),
        verdict_token,
        int(vocab.eos_id),
    ]

    input_ids = np.full((int(seq_length),), fill_value=int(vocab.pad_id), dtype=np.int64)
    labels = np.full((int(seq_length),), fill_value=int(vocab.ignore_index), dtype=np.int64)
    used = min(len(tokens), int(seq_length))
    input_ids[:used] = np.asarray(tokens[:used], dtype=np.int64)
    if used > 1:
        labels[: (used - 1)] = input_ids[1:used]
    return input_ids, labels, np.asarray(score, dtype=np.float32)


def _make_dataset_arrays(cfg: DataConfig, vocab: Vocab):
    rng = np.random.default_rng(int(cfg.seed))
    n = int(cfg.num_samples)
    t = int(cfg.seq_length)

    input_ids = np.empty((n, t), dtype=np.int64)
    labels = np.empty((n, t), dtype=np.int64)
    judge_targets = np.empty((n,), dtype=np.float32)

    for idx in range(n):
        prompt_value = int(rng.integers(0, int(vocab.base_vocab_size)))
        reference_value = int(rng.integers(0, int(vocab.base_vocab_size)))
        if bool(rng.integers(0, 2)):
            offset = int(rng.integers(-2, 3))
            candidate_value = (reference_value + offset) % int(vocab.base_vocab_size)
        else:
            candidate_value = int((reference_value + rng.integers(8, 16)) % int(vocab.base_vocab_size))

        ids_i, labels_i, score_i = _build_sample(
            prompt_value=prompt_value,
            reference_value=reference_value,
            candidate_value=int(candidate_value),
            vocab=vocab,
            seq_length=t,
        )
        input_ids[idx] = ids_i
        labels[idx] = labels_i
        judge_targets[idx] = score_i
    return input_ids, labels, judge_targets


class SyntheticLlmJudgeDataset:
    def __init__(self, *, input_ids: np.ndarray, labels: np.ndarray, judge_targets: np.ndarray, vocab: Vocab) -> None:
        self.input_ids = input_ids
        self.labels = labels
        self.judge_targets = judge_targets
        self.vocab = vocab

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int):
        import torch

        ids = torch.tensor(self.input_ids[int(idx)], dtype=torch.long)
        labels = torch.tensor(self.labels[int(idx)], dtype=torch.long)
        judge_target = torch.tensor(self.judge_targets[int(idx)], dtype=torch.float32)
        return {
            "input_ids": ids,
            "attention_mask": (ids != int(self.vocab.pad_id)).to(torch.float32),
            "labels": labels,
            "judge_targets": judge_target,
        }


def get_dataloaders(cfg: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    if int(cfg.seq_length) < 9:
        raise ValueError("seq_length must be >= 9 for synthetic LLM judge sequences")

    vocab = Vocab(base_vocab_size=int(cfg.base_vocab_size))
    input_ids, labels, judge_targets = _make_dataset_arrays(cfg, vocab)
    ds = SyntheticLlmJudgeDataset(
        input_ids=input_ids,
        labels=labels,
        judge_targets=judge_targets,
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
            "judge_targets": torch.stack([row["judge_targets"] for row in batch], dim=0),
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


__all__ = ["DataConfig", "SyntheticLlmJudgeDataset", "Vocab", "get_dataloaders"]
