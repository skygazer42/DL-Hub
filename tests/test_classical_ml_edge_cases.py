import numpy as np
import pytest

from ml_algorithms.python.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from ml_algorithms.python.kmeans import KMeans
from ml_algorithms.python.knn import KNNClassifier, KNNRegressor


def test_knn_supports_using_every_training_sample_as_a_neighbor() -> None:
    x = np.array([[0.0], [2.0]])

    classifier = KNNClassifier(k=2).fit(x, np.array([0, 1]))
    regressor = KNNRegressor(k=2).fit(x, np.array([1.0, 3.0]))

    assert classifier.predict([[1.0]]).item() == 0
    assert regressor.predict([[1.0]]).item() == pytest.approx(2.0)


@pytest.mark.parametrize("estimator", [KNNClassifier(k=0), KNNRegressor(k=3)])
def test_knn_rejects_invalid_neighbor_count(estimator) -> None:
    with pytest.raises(ValueError, match="k must be"):
        estimator.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))


@pytest.mark.parametrize("estimator", [KNNClassifier(), KNNRegressor()])
def test_knn_has_clear_fit_and_shape_contracts(estimator) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        estimator.predict(np.array([[0.0]]))

    with pytest.raises(ValueError, match="same number of samples"):
        estimator.fit(np.array([[0.0], [1.0]]), np.array([0]))


def test_kmeans_rejects_invalid_iteration_count_before_sampling() -> None:
    with pytest.raises(ValueError, match="max_iter"):
        KMeans(n_clusters=1, max_iter=0).fit(np.array([[0.0], [1.0]]))


def test_kmeans_is_reproducible_and_constant_features_stay_finite() -> None:
    x = np.ones((4, 2), dtype=np.float64)

    first = KMeans(n_clusters=2, random_state=7).fit(x)
    second = KMeans(n_clusters=2, random_state=7).fit(x)

    np.testing.assert_array_equal(first.labels_, second.labels_)
    np.testing.assert_array_equal(first.cluster_centers_, second.cluster_centers_)
    assert np.all(np.isfinite(first.cluster_centers_))


def test_kmeans_labels_match_final_centers_when_iteration_budget_is_exhausted() -> None:
    x = np.array([[0.0], [2.0], [3.0], [10.0]])

    model = KMeans(n_clusters=2, max_iter=1, random_state=1).fit(x)

    np.testing.assert_array_equal(model.labels_, model.predict(x))


@pytest.mark.parametrize(
    ("estimator", "expected"),
    [
        (DecisionTreeClassifier(max_depth=2), np.array([0, 0])),
        (DecisionTreeRegressor(max_depth=2), np.array([2.0, 2.0])),
    ],
)
def test_decision_tree_constant_features_produce_finite_leaf_predictions(
    estimator, expected
) -> None:
    x = np.ones((2, 1), dtype=np.float64)
    y = np.array([0, 0]) if isinstance(estimator, DecisionTreeClassifier) else np.array([1.0, 3.0])

    predictions = estimator.fit(x, y).predict(x)

    np.testing.assert_array_equal(predictions, expected)


def test_decision_tree_rejects_unfitted_and_mismatched_data() -> None:
    tree = DecisionTreeClassifier()
    with pytest.raises(RuntimeError, match="not fitted"):
        tree.predict(np.array([[0.0]]))

    with pytest.raises(ValueError, match="same number of samples"):
        tree.fit(np.array([[0.0], [1.0]]), np.array([0]))

    with pytest.raises(ValueError, match="min_samples_split"):
        DecisionTreeClassifier(min_samples_split=1).fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
