"""Discriminant analysis classifiers in NumPy (LDA / QDA)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LDAClassifier:
    """Linear Discriminant Analysis classifier.

    Assumes class-conditional Gaussians with a shared covariance matrix.
    """

    reg: float = 1e-6

    def fit(self, x: np.ndarray, y: np.ndarray) -> LDAClassifier:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y).ravel()
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of labels")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")

        reg = float(self.reg)
        if reg < 0.0:
            raise ValueError("reg must be >= 0")

        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("Need at least 2 classes to fit LDAClassifier")

        n_samples, n_features = x.shape
        n_classes = int(classes.size)

        means = np.empty((n_classes, n_features), dtype=np.float64)
        priors = np.empty((n_classes,), dtype=np.float64)
        scatter = np.zeros((n_features, n_features), dtype=np.float64)

        for idx, cls in enumerate(classes):
            x_cls = x[y == cls]
            if x_cls.size == 0:
                raise ValueError("Encountered an empty class during fit")
            mean = x_cls.mean(axis=0)
            means[idx] = mean
            priors[idx] = x_cls.shape[0] / n_samples
            centered = x_cls - mean
            scatter += centered.T @ centered

        denom = float(max(1, n_samples - n_classes))
        covariance = scatter / denom
        if reg != 0.0:
            covariance = covariance + reg * np.eye(n_features, dtype=np.float64)

        inv_cov_means = np.linalg.solve(covariance, means.T)
        intercept = -0.5 * np.sum(means * inv_cov_means.T, axis=1) + np.log(priors)

        self.classes_ = classes
        self.means_ = means
        self.priors_ = priors
        self.covariance_ = covariance
        self.inv_cov_means_ = inv_cov_means
        self.intercept_ = intercept
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        scores = x @ self.inv_cov_means_ + self.intercept_
        return self.classes_[np.argmax(scores, axis=1)]


@dataclass
class QDAClassifier:
    """Quadratic Discriminant Analysis classifier.

    Assumes class-conditional Gaussians with class-specific covariance matrices.
    """

    reg: float = 1e-6

    def fit(self, x: np.ndarray, y: np.ndarray) -> QDAClassifier:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y).ravel()
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of labels")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")

        reg = float(self.reg)
        if reg < 0.0:
            raise ValueError("reg must be >= 0")

        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("Need at least 2 classes to fit QDAClassifier")

        n_samples, n_features = x.shape
        n_classes = int(classes.size)

        means = np.empty((n_classes, n_features), dtype=np.float64)
        priors = np.empty((n_classes,), dtype=np.float64)
        covariances = np.empty((n_classes, n_features, n_features), dtype=np.float64)
        log_dets = np.empty((n_classes,), dtype=np.float64)

        for idx, cls in enumerate(classes):
            x_cls = x[y == cls]
            if x_cls.size == 0:
                raise ValueError("Encountered an empty class during fit")
            mean = x_cls.mean(axis=0)
            means[idx] = mean
            priors[idx] = x_cls.shape[0] / n_samples

            centered = x_cls - mean
            denom = float(max(1, x_cls.shape[0] - 1))
            cov = (centered.T @ centered) / denom
            if reg != 0.0:
                cov = cov + reg * np.eye(n_features, dtype=np.float64)

            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                raise ValueError("Covariance matrix must be positive definite after regularization")

            covariances[idx] = cov
            log_dets[idx] = float(logdet)

        self.classes_ = classes
        self.means_ = means
        self.priors_ = priors
        self.log_priors_ = np.log(priors)
        self.covariances_ = covariances
        self.log_dets_ = log_dets
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")

        n_samples = x.shape[0]
        n_classes = int(self.classes_.size)
        scores = np.empty((n_samples, n_classes), dtype=np.float64)

        for idx in range(n_classes):
            mean = self.means_[idx]
            cov = self.covariances_[idx]
            centered = x - mean
            solved = np.linalg.solve(cov, centered.T).T
            quad = np.sum(centered * solved, axis=1)
            scores[:, idx] = -0.5 * (self.log_dets_[idx] + quad) + self.log_priors_[idx]

        return self.classes_[np.argmax(scores, axis=1)]
