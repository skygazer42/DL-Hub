"""Decision tree classifier and regressor (CART-style) in NumPy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class Node:
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    value: Optional[float] = None


class DecisionTreeBase:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[Node] = None

    def _best_split(self, x: np.ndarray, y: np.ndarray, criterion: Callable[[np.ndarray], float]):
        best_feature = None
        best_threshold = None
        best_gain = -np.inf
        current_impurity = criterion(y)
        for feature in range(x.shape[1]):
            thresholds = np.unique(x[:, feature])
            for threshold in thresholds:
                left_mask = x[:, feature] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                left_impurity = criterion(y[left_mask])
                right_impurity = criterion(y[right_mask])
                weighted_impurity = (
                    left_mask.mean() * left_impurity + right_mask.mean() * right_impurity
                )
                gain = current_impurity - weighted_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int):
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return Node(value=self._leaf_value(y))
        feature, threshold = self._best_split(x, y, self._criterion)
        if feature is None:
            return Node(value=self._leaf_value(y))
        left_mask = x[:, feature] <= threshold
        right_mask = ~left_mask
        return Node(
            feature_index=feature,
            threshold=float(threshold),
            left=self._build_tree(x[left_mask], y[left_mask], depth + 1),
            right=self._build_tree(x[right_mask], y[right_mask], depth + 1),
        )

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        self.root = self._build_tree(x, y, 0)
        return self

    def _predict_one(self, node: Node, sample: np.ndarray):
        if node.value is not None:
            return node.value
        if sample[node.feature_index] <= node.threshold:
            return self._predict_one(node.left, sample)
        return self._predict_one(node.right, sample)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return np.array([self._predict_one(self.root, sample) for sample in x])

    def _criterion(self, y: np.ndarray) -> float:
        raise NotImplementedError

    def _leaf_value(self, y: np.ndarray):
        raise NotImplementedError


class DecisionTreeClassifier(DecisionTreeBase):
    def _criterion(self, y: np.ndarray) -> float:
        _, counts = np.unique(y, return_counts=True)
        prob = counts / counts.sum()
        return 1.0 - np.sum(prob ** 2)

    def _leaf_value(self, y: np.ndarray):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]


class DecisionTreeRegressor(DecisionTreeBase):
    def _criterion(self, y: np.ndarray) -> float:
        return float(np.var(y))

    def _leaf_value(self, y: np.ndarray):
        return float(np.mean(y))
