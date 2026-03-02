"""Naive Bayes classifiers in NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianNB:
    def fit(self, x: np.ndarray, y: np.ndarray) -> GaussianNB:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.mean_ = {}
        self.var_ = {}
        self.prior_ = {}
        for cls in self.classes_:
            x_cls = x[y == cls]
            self.mean_[cls] = x_cls.mean(axis=0)
            self.var_[cls] = x_cls.var(axis=0) + 1e-9
            self.prior_[cls] = x_cls.shape[0] / x.shape[0]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        log_probs = []
        for cls in self.classes_:
            mean = self.mean_[cls]
            var = self.var_[cls]
            log_likelihood = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            log_likelihood -= 0.5 * np.sum(((x - mean) ** 2) / var, axis=1)
            log_prior = np.log(self.prior_[cls])
            log_probs.append(log_prior + log_likelihood)
        log_probs = np.vstack(log_probs).T
        return self.classes_[np.argmax(log_probs, axis=1)]


@dataclass
class MultinomialNB:
    alpha: float = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> MultinomialNB:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.class_count_ = {}
        self.feature_log_prob_ = {}
        for cls in self.classes_:
            x_cls = x[y == cls]
            class_count = x_cls.shape[0]
            self.class_count_[cls] = class_count
            feature_count = x_cls.sum(axis=0) + self.alpha
            self.feature_log_prob_[cls] = np.log(feature_count / feature_count.sum())
        self.class_log_prior_ = {
            cls: np.log(count / x.shape[0]) for cls, count in self.class_count_.items()
        }
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        log_probs = []
        for cls in self.classes_:
            log_likelihood = x @ self.feature_log_prob_[cls]
            log_probs.append(self.class_log_prior_[cls] + log_likelihood)
        log_probs = np.vstack(log_probs).T
        return self.classes_[np.argmax(log_probs, axis=1)]
