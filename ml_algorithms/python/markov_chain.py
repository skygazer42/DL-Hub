"""Markov chain model in NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MarkovChain:
    """First-order Markov chain for discrete (integer) states.

    Learns a transition probability matrix P(next_state | current_state) from sequences.
    """

    alpha: float = 0.0

    def fit(self, sequences: object) -> MarkovChain:
        seqs: list[np.ndarray]

        if isinstance(sequences, np.ndarray):
            if sequences.ndim == 1:
                seqs = [sequences]
            elif sequences.ndim == 2:
                seqs = [row for row in sequences]
            else:
                raise ValueError("sequences must be 1D/2D array or an iterable of 1D arrays")
        else:
            seqs = list(sequences)  # type: ignore[arg-type]

        if not seqs:
            raise ValueError("sequences must be non-empty")

        observed: list[np.ndarray] = []
        n_transitions = 0
        for seq in seqs:
            arr = np.asarray(seq).ravel()
            if arr.size == 0:
                continue
            arr = arr.astype(int, copy=False)
            observed.append(arr)
            n_transitions += max(0, int(arr.size) - 1)

        if not observed:
            raise ValueError("sequences must contain at least one non-empty sequence")
        if n_transitions == 0:
            raise ValueError("Need at least one transition (sequence length >= 2)")

        alpha = float(self.alpha)
        if alpha < 0.0:
            raise ValueError("alpha must be >= 0")

        states = np.unique(np.concatenate(observed)).astype(int, copy=False)
        n_states = int(states.size)
        counts = np.zeros((n_states, n_states), dtype=np.float64)

        for seq in observed:
            if seq.size < 2:
                continue
            idx = np.searchsorted(states, seq)
            src = idx[:-1]
            dst = idx[1:]
            np.add.at(counts, (src, dst), 1.0)

        counts_smooth = counts + alpha
        row_sums = counts_smooth.sum(axis=1, keepdims=True)
        transition = np.divide(
            counts_smooth,
            row_sums,
            out=np.zeros_like(counts_smooth),
            where=row_sums != 0.0,
        )

        self.states_ = states
        self.transition_counts_ = counts
        self.transition_matrix_ = transition
        return self

    def predict_next(self, state: int) -> int:
        state_int = int(state)
        idx = int(np.searchsorted(self.states_, state_int))
        if idx >= int(self.states_.size) or int(self.states_[idx]) != state_int:
            raise ValueError(f"Unknown state: {state!r}")

        probs = self.transition_matrix_[idx]
        if not np.any(probs):
            raise ValueError(f"State {state_int} has no outgoing transitions")
        next_idx = int(np.argmax(probs))
        return int(self.states_[next_idx])
