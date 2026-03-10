"""Hidden Markov Model (HMM) with categorical emissions in NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _logsumexp(a: np.ndarray, *, axis: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    a_max = np.max(a, axis=axis, keepdims=True)

    # When all entries along the axis are -inf, a_max is -inf and (a - a_max) is NaN.
    # In that case exp should contribute 0 to the sum.
    stable = np.where(np.isfinite(a_max), np.exp(a - a_max), 0.0)
    s = np.sum(stable, axis=axis, keepdims=True)
    out = np.where(s > 0.0, a_max + np.log(s), -np.inf)
    return np.squeeze(out, axis=axis)


def _row_normalize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    row_sums = a.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("All rows must sum to a positive value.")
    return a / row_sums


def _safe_log(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return np.where(a > 0.0, np.log(a), -np.inf)


@dataclass
class CategoricalHMM:
    startprob: np.ndarray
    transmat: np.ndarray
    emissionprob: np.ndarray

    def fit(
        self,
        observations: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
        *,
        n_iter: int = 10,
        tol: float = 1e-4,
    ) -> CategoricalHMM:
        startprob = np.asarray(self.startprob, dtype=np.float64).ravel()
        transmat = np.asarray(self.transmat, dtype=np.float64)
        emissionprob = np.asarray(self.emissionprob, dtype=np.float64)

        if startprob.ndim != 1:
            raise ValueError("startprob must be a 1D array of shape (n_states,)")
        if transmat.ndim != 2:
            raise ValueError("transmat must be a 2D array of shape (n_states, n_states)")
        if emissionprob.ndim != 2:
            raise ValueError("emissionprob must be a 2D array of shape (n_states, n_observations)")

        n_states = int(startprob.size)
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        if transmat.shape != (n_states, n_states):
            raise ValueError("transmat shape must match (n_states, n_states)")
        if emissionprob.shape[0] != n_states:
            raise ValueError("emissionprob must have n_states rows")
        n_observations = int(emissionprob.shape[1])
        if n_observations < 1:
            raise ValueError("n_observations must be >= 1")

        if np.any(startprob < 0.0) or np.any(transmat < 0.0) or np.any(emissionprob < 0.0):
            raise ValueError("Probabilities must be non-negative")

        start_sum = float(startprob.sum())
        if start_sum <= 0.0:
            raise ValueError("startprob must sum to a positive value")

        self.startprob_ = startprob / start_sum
        self.transmat_ = _row_normalize(transmat)
        self.emissionprob_ = _row_normalize(emissionprob)

        self.n_states_ = n_states
        self.n_observations_ = n_observations

        self.log_startprob_ = _safe_log(self.startprob_)
        self.log_transmat_ = _safe_log(self.transmat_)
        self.log_emissionprob_ = _safe_log(self.emissionprob_)

        if observations is None:
            return self

        if int(n_iter) < 1:
            raise ValueError("n_iter must be >= 1")
        if float(tol) < 0.0:
            raise ValueError("tol must be >= 0")

        sequences: list[np.ndarray] = self._as_sequences(observations)
        sequences = [self._validate_observations(seq) for seq in sequences]

        eps = 1e-12
        prev_ll = -np.inf
        for _ in range(int(n_iter)):
            start_counts = np.zeros((self.n_states_,), dtype=np.float64)
            trans_counts = np.zeros((self.n_states_, self.n_states_), dtype=np.float64)
            emission_counts = np.zeros((self.n_states_, self.n_observations_), dtype=np.float64)

            total_ll = 0.0
            for seq in sequences:
                log_alpha, ll = self._forward_log(seq)
                log_beta = self._backward_log(seq)
                total_ll += ll

                log_gamma = log_alpha + log_beta - ll
                gamma = np.exp(log_gamma)
                row_sums = gamma.sum(axis=1, keepdims=True)
                gamma = np.where(row_sums > 0.0, gamma / row_sums, 1.0 / float(self.n_states_))

                start_counts += gamma[0]

                for t in range(int(seq.size) - 1):
                    next_obs = int(seq[t + 1])
                    log_xi = (
                        log_alpha[t][:, None]
                        + self.log_transmat_
                        + self.log_emissionprob_[:, next_obs][None, :]
                        + log_beta[t + 1][None, :]
                        - ll
                    )
                    xi = np.exp(log_xi)
                    xi_sum = float(xi.sum())
                    if xi_sum > 0.0:
                        xi /= xi_sum
                    else:
                        xi.fill(1.0 / float(self.n_states_ * self.n_states_))
                    trans_counts += xi

                for t, symbol in enumerate(seq):
                    emission_counts[:, int(symbol)] += gamma[t]

            start_counts += eps
            trans_counts += eps
            emission_counts += eps

            self.startprob_ = start_counts / float(start_counts.sum())
            self.transmat_ = _row_normalize(trans_counts)
            self.emissionprob_ = _row_normalize(emission_counts)

            self.log_startprob_ = _safe_log(self.startprob_)
            self.log_transmat_ = _safe_log(self.transmat_)
            self.log_emissionprob_ = _safe_log(self.emissionprob_)

            if np.isfinite(prev_ll) and abs(total_ll - prev_ll) < float(tol):
                break
            prev_ll = total_ll
        return self

    def _as_sequences(
        self, observations: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]
    ) -> list[np.ndarray]:
        if isinstance(observations, np.ndarray):
            obs = np.asarray(observations, dtype=int)
            if obs.ndim == 1:
                return [obs]
            if obs.ndim == 2:
                return [obs[i] for i in range(int(obs.shape[0]))]
            raise ValueError("observations must be a 1D array, 2D array, or list of 1D arrays")

        if isinstance(observations, list | tuple):
            if len(observations) < 1:
                raise ValueError("observations must contain at least one sequence")
            return [np.asarray(seq, dtype=int) for seq in observations]

        raise TypeError("observations must be a numpy array or a list of numpy arrays")

    def _validate_observations(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations, dtype=int).ravel()
        if obs.size < 1:
            raise ValueError("observations must be non-empty")
        if np.min(obs) < 0 or np.max(obs) >= self.n_observations_:
            raise ValueError("observations must be ints in [0, n_observations)")
        return obs

    def _forward_log(self, observations: np.ndarray) -> tuple[np.ndarray, float]:
        obs = np.asarray(observations, dtype=int).ravel()
        t_max = int(obs.size)

        log_alpha = np.empty((t_max, self.n_states_), dtype=np.float64)
        log_alpha[0] = self.log_startprob_ + self.log_emissionprob_[:, int(obs[0])]
        for t in range(1, t_max):
            emission = self.log_emissionprob_[:, int(obs[t])]
            log_alpha[t] = emission + _logsumexp(
                log_alpha[t - 1][:, None] + self.log_transmat_,
                axis=0,
            )
        ll = float(_logsumexp(log_alpha[-1], axis=0))
        return log_alpha, ll

    def _backward_log(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations, dtype=int).ravel()
        t_max = int(obs.size)

        log_beta = np.empty((t_max, self.n_states_), dtype=np.float64)
        log_beta[-1] = 0.0
        for t in range(t_max - 2, -1, -1):
            next_terms = self.log_emissionprob_[:, int(obs[t + 1])] + log_beta[t + 1]
            log_beta[t] = _logsumexp(self.log_transmat_ + next_terms[None, :], axis=1)
        return log_beta

    def score(self, observations: np.ndarray) -> float:
        obs = self._validate_observations(observations)

        _, ll = self._forward_log(obs)
        return ll

    def predict(self, observations: np.ndarray) -> np.ndarray:
        obs = self._validate_observations(observations)

        t_max = int(obs.size)
        delta = self.log_startprob_ + self.log_emissionprob_[:, int(obs[0])]
        psi = np.zeros((t_max, self.n_states_), dtype=int)

        for t in range(1, t_max):
            scores = delta[:, None] + self.log_transmat_
            psi[t] = np.argmax(scores, axis=0)
            delta = self.log_emissionprob_[:, int(obs[t])] + np.max(scores, axis=0)

        states = np.empty((t_max,), dtype=int)
        states[-1] = int(np.argmax(delta))
        for t in range(t_max - 2, -1, -1):
            states[t] = int(psi[t + 1, states[t + 1]])
        return states
