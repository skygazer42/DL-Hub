import numpy as np

from ml_algorithms.python.naive_bayes import BernoulliNB


def test_bernoulli_nb_classifies_obvious_binary_patterns() -> None:
    x = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 1],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 1, 1, 1])

    model = BernoulliNB(alpha=1.0).fit(x, y)
    preds = model.predict(x)

    accuracy = (preds == y).mean()
    assert accuracy >= 0.95


def test_bernoulli_nb_thresholds_inputs_as_binary() -> None:
    # Bernoulli NB treats features as binary via (x > 0).
    x = np.array(
        [
            [2.0, 0.2, -1.0],
            [0.1, 0.0, 0.0],
            [-0.3, -2.0, 3.0],
            [-1.0, 0.0, 1.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = BernoulliNB(alpha=1.0).fit(x, y)
    preds = model.predict(x)
    assert np.array_equal(preds, y)
