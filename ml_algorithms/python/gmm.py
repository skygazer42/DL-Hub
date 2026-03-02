"""Gaussian Mixture Model with diagonal covariance in NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianMixture:
    n_components: int = 3
    max_iter: int = 100
    tol: float = 1e-4
    random_state: int | None = None

    def fit(self, x: np.ndarray) -> GaussianMixture:
        rng = np.random.default_rng(self.random_state)
        x = np.asarray(x, dtype=np.float64)
        n_samples, n_features = x.shape
        indices = rng.choice(n_samples, self.n_components, replace=False)
        self.means_ = x[indices]
        self.covariances_ = np.ones((self.n_components, n_features))
        self.weights_ = np.full(self.n_components, 1.0 / self.n_components)
        log_likelihood_prev = None
        for _ in range(self.max_iter):
            resp = self._estimate_responsibilities(x)
            effective_n = resp.sum(axis=0)
            self.weights_ = effective_n / n_samples
            self.means_ = (resp.T @ x) / effective_n[:, None]
            diff = x[:, None, :] - self.means_[None, :, :]
            self.covariances_ = (resp[:, :, None] * diff**2).sum(axis=0) / effective_n[:, None]
            log_likelihood = np.sum(np.log(resp.sum(axis=1) + 1e-12))
            if (
                log_likelihood_prev is not None
                and abs(log_likelihood - log_likelihood_prev) < self.tol
            ):
                break
            log_likelihood_prev = log_likelihood
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        resp = self._estimate_responsibilities(np.asarray(x, dtype=np.float64))
        return np.argmax(resp, axis=1)

    def _estimate_responsibilities(self, x: np.ndarray) -> np.ndarray:
        probs = []
        for weight, mean, cov in zip(self.weights_, self.means_, self.covariances_):
            var = np.maximum(cov, 1e-6)
            log_prob = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            log_prob -= 0.5 * np.sum(((x - mean) ** 2) / var, axis=1)
            probs.append(np.log(weight + 1e-12) + log_prob)
        log_probs = np.vstack(probs).T
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        return probs / probs.sum(axis=1, keepdims=True)
