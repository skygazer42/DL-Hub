import numpy as np

from ml_algorithms.python.knn import KNNClassifier, KNNRegressor
from ml_algorithms.python.naive_bayes import GaussianNB


def test_gaussian_nb_classifies_separated_gaussians() -> None:
    rng = np.random.default_rng(0)
    x0 = rng.normal(loc=(-2.0, -2.0), scale=0.6, size=(250, 2))
    x1 = rng.normal(loc=(2.0, 2.0), scale=0.6, size=(250, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * x0.shape[0] + [1] * x1.shape[0])

    model = GaussianNB().fit(x, y)
    preds = model.predict(x)

    accuracy = (preds == y).mean()
    assert accuracy > 0.9


def test_knn_classifier_predicts_nearest_majority_label() -> None:
    x_train = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    clf = KNNClassifier(k=3).fit(x_train, y_train)
    preds = clf.predict(np.array([[1.5], [10.5]]))

    assert np.array_equal(preds, np.array([0, 1]))


def test_knn_regressor_predicts_mean_of_neighbors() -> None:
    x_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([0.0, 1.0, 2.0, 3.0])

    reg = KNNRegressor(k=2).fit(x_train, y_train)
    pred = reg.predict(np.array([[1.2]]))[0]

    assert pred == np.mean([1.0, 2.0])
