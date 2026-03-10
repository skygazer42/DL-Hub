"""AdaBoost classifier (binary) using decision stumps in NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _stump_predict(
    x: np.ndarray,
    *,
    feature_index: int,
    threshold: float,
    polarity: int,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    col = x[:, feature_index]
    pred = np.where(col >= threshold, polarity, -polarity)
    return pred.astype(np.float64, copy=False)


def _find_best_stump(
    x: np.ndarray,
    y_pm: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[int, float, int, np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    y_pm = np.asarray(y_pm, dtype=np.float64).ravel()
    sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()

    n_samples, n_features = x.shape
    best_feature = 0
    best_threshold = 0.0
    best_polarity = 1
    best_error = np.inf

    for feature_index in range(int(n_features)):
        col = x[:, feature_index]
        order = np.argsort(col, kind="mergesort")
        xs = col[order]
        ys = y_pm[order]
        ws = sample_weight[order]

        pos = ws * (ys > 0.0)
        neg = ws * (ys < 0.0)

        cum_pos = np.cumsum(pos)
        cum_neg = np.cumsum(neg)

        total_pos = float(cum_pos[-1])
        total_neg = float(cum_neg[-1])

        # Candidate 1: constant +1 predictions (threshold below minimum).
        err_const_pos = total_neg
        thr_const_pos = float(np.nextafter(xs[0], -np.inf))
        if err_const_pos < best_error:
            best_error = err_const_pos
            best_feature = feature_index
            best_threshold = thr_const_pos
            best_polarity = 1

        # Candidate 2: constant -1 predictions (threshold above maximum).
        err_const_neg = total_pos
        thr_const_neg = float(np.nextafter(xs[-1], np.inf))
        if err_const_neg < best_error:
            best_error = err_const_neg
            best_feature = feature_index
            best_threshold = thr_const_neg
            best_polarity = 1

        if n_samples < 2:
            continue

        valid_split = xs[:-1] != xs[1:]
        if not np.any(valid_split):
            continue

        split_idx = np.nonzero(valid_split)[0]

        left_pos = cum_pos[split_idx]
        left_neg = cum_neg[split_idx]
        right_pos = total_pos - left_pos
        right_neg = total_neg - left_neg

        # polarity = +1 => predict -1 for x < thr, +1 for x >= thr
        err_pol_pos = left_pos + right_neg
        best_local_pos = int(np.argmin(err_pol_pos))
        err_pos = float(err_pol_pos[best_local_pos])

        # polarity = -1 => predict +1 for x < thr, -1 for x >= thr
        err_pol_neg = left_neg + right_pos
        best_local_neg = int(np.argmin(err_pol_neg))
        err_neg = float(err_pol_neg[best_local_neg])

        if err_pos < best_error:
            i = int(split_idx[best_local_pos])
            best_error = err_pos
            best_feature = feature_index
            best_threshold = float(0.5 * (xs[i] + xs[i + 1]))
            best_polarity = 1

        if err_neg < best_error:
            i = int(split_idx[best_local_neg])
            best_error = err_neg
            best_feature = feature_index
            best_threshold = float(0.5 * (xs[i] + xs[i + 1]))
            best_polarity = -1

    pred = _stump_predict(
        x,
        feature_index=best_feature,
        threshold=best_threshold,
        polarity=best_polarity,
    )
    error = float(np.sum(sample_weight[pred != y_pm]))
    return best_feature, best_threshold, best_polarity, pred, error


@dataclass
class AdaBoostClassifier:
    n_estimators: int = 50
    learning_rate: float = 1.0
    random_state: int | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> AdaBoostClassifier:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        y = np.asarray(y)
        y = y.ravel()
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")

        unique = np.unique(y)
        if unique.size == 0:
            raise ValueError("y must be non-empty")
        if not np.all(np.isin(unique, np.array([0, 1]))):
            raise ValueError("This implementation supports binary labels {0, 1} only.")

        if int(self.n_estimators) < 1:
            raise ValueError("n_estimators must be >= 1")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be > 0")

        self.classes_ = np.array([0, 1], dtype=int)
        self.base_rate_ = float(np.mean(y.astype(np.float64)))
        self.majority_class_ = int(self.base_rate_ >= 0.5)

        y_pm = np.where(y == 1, 1.0, -1.0).astype(np.float64, copy=False)
        n_samples = int(x.shape[0])

        sample_weight = np.full(n_samples, 1.0 / float(n_samples), dtype=np.float64)

        self.estimators_: list[tuple[int, float, int]] = []
        self.estimator_weights_: list[float] = []

        _ = np.random.default_rng(self.random_state)

        for _ in range(int(self.n_estimators)):
            feature_index, threshold, polarity, pred, err = _find_best_stump(x, y_pm, sample_weight)

            if err >= 0.5:
                break

            err = float(np.clip(err, 1e-12, 1.0 - 1e-12))
            alpha = 0.5 * float(np.log((1.0 - err) / err)) * float(self.learning_rate)

            sample_weight *= np.exp(-alpha * y_pm * pred)
            weight_sum = float(sample_weight.sum())
            if weight_sum <= 0.0 or not np.isfinite(weight_sum):
                break
            sample_weight /= weight_sum

            self.estimators_.append((int(feature_index), float(threshold), int(polarity)))
            self.estimator_weights_.append(float(alpha))

            if err <= 1e-12:
                break

        self.estimator_weights_ = np.asarray(self.estimator_weights_, dtype=np.float64)
        return self

    def _decision_function(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if len(self.estimators_) == 0:
            return np.zeros((x.shape[0],), dtype=np.float64)

        scores = np.zeros((x.shape[0],), dtype=np.float64)
        for (feature_index, threshold, polarity), alpha in zip(
            self.estimators_, self.estimator_weights_, strict=True
        ):
            scores += float(alpha) * _stump_predict(
                x,
                feature_index=int(feature_index),
                threshold=float(threshold),
                polarity=int(polarity),
            )
        return scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self._decision_function(x)
        if len(self.estimators_) == 0:
            return np.full(scores.shape, self.majority_class_, dtype=int)
        return (scores >= 0.0).astype(int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = self._decision_function(x)
        if len(self.estimators_) == 0:
            p1 = np.full(scores.shape, self.base_rate_, dtype=np.float64)
        else:
            z = np.clip(2.0 * scores, -500.0, 500.0)
            p1 = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p1, p1])
