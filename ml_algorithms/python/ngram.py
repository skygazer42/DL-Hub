"""N-gram language model in NumPy."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class NGramLanguageModel:
    n: int = 2
    alpha: float = 1.0

    fitted_: bool = field(init=False, default=False)

    def fit(self, sequences: Iterable[Sequence[int]] | np.ndarray) -> NGramLanguageModel:
        n = int(self.n)
        if n < 1:
            raise ValueError("n must be >= 1")
        alpha = float(self.alpha)
        if alpha < 0.0:
            raise ValueError("alpha must be >= 0")

        if isinstance(sequences, np.ndarray):
            if sequences.ndim == 1:
                sequences_list = [sequences]
            elif sequences.ndim == 2:
                sequences_list = [row for row in sequences]
            else:
                raise ValueError("sequences must be 1D or 2D when passing a NumPy array")
        else:
            sequences_list = list(sequences)

        seq_arrays: list[np.ndarray] = []
        for seq in sequences_list:
            arr = np.asarray(seq, dtype=np.int64).ravel()
            if arr.size == 0:
                continue
            if np.any(arr < 0):
                raise ValueError("tokens must be non-negative integers")
            seq_arrays.append(arr)

        if not seq_arrays:
            raise ValueError("Need at least one non-empty sequence")

        vocab = np.unique(np.concatenate(seq_arrays, axis=0))
        if vocab.size == 0:
            raise ValueError("Need at least one token to fit")

        self.vocab_ = vocab.astype(np.int64, copy=False)
        self.vocab_size_ = int(self.vocab_.size)
        self.token_to_id_ = {int(token): int(i) for i, token in enumerate(self.vocab_)}

        context_len = n - 1
        self.context_len_ = int(context_len)
        self.context_counts_: dict[tuple[int, ...], np.ndarray] = {}
        self.context_totals_: dict[tuple[int, ...], float] = {}

        for seq in seq_arrays:
            ids = np.array([self.token_to_id_[int(t)] for t in seq], dtype=np.int64)
            if ids.size < n:
                continue
            for i in range(context_len, int(ids.size)):
                key = tuple(ids[i - context_len : i].tolist())
                next_id = int(ids[i])
                counts = self.context_counts_.get(key)
                if counts is None:
                    counts = np.zeros(self.vocab_size_, dtype=np.float64)
                    self.context_counts_[key] = counts
                    self.context_totals_[key] = 0.0
                counts[next_id] += 1.0
                self.context_totals_[key] += 1.0

        self.fitted_ = True
        return self

    def predict_next_proba(self, context: Sequence[int] | np.ndarray | int | None) -> np.ndarray:
        if not self.fitted_:
            raise ValueError("Model is not fitted. Call fit(...) first.")

        if context is None:
            context_tokens = np.array([], dtype=np.int64)
        else:
            context_tokens = np.asarray(context, dtype=np.int64).ravel()

        if int(context_tokens.size) != int(self.context_len_):
            raise ValueError(f"context must have length {self.context_len_}")

        try:
            key = tuple(self.token_to_id_[int(t)] for t in context_tokens.tolist())
        except KeyError:
            return np.full((self.vocab_size_,), 1.0 / float(self.vocab_size_), dtype=np.float64)

        counts = self.context_counts_.get(key)
        total = float(self.context_totals_.get(key, 0.0))
        if counts is None:
            counts = np.zeros((self.vocab_size_,), dtype=np.float64)
            total = 0.0

        alpha = float(self.alpha)
        denom = total + alpha * float(self.vocab_size_)
        if denom <= 0.0:
            return np.full((self.vocab_size_,), 1.0 / float(self.vocab_size_), dtype=np.float64)

        return (counts + alpha) / denom

    def predict_next(self, context: Sequence[int] | np.ndarray | int | None) -> int:
        proba = self.predict_next_proba(context)
        next_id = int(np.argmax(proba))
        return int(self.vocab_[next_id])
