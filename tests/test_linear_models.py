import numpy as np

from ml_algorithms.python.linear_models import LinearRegression, LogisticRegression


def test_linear_regression_fits_simple_relationship() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(256, 2))
    y = 0.5 + 3.0 * x[:, 0] - 2.0 * x[:, 1] + rng.normal(scale=0.01, size=x.shape[0])

    model = LinearRegression(learning_rate=0.05, epochs=2000).fit(x, y)
    preds = model.predict(x)

    mse = np.mean((preds - y) ** 2)
    assert mse < 1e-3


def test_logistic_regression_separates_linearly_separable_data() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 2))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)

    clf = LogisticRegression(learning_rate=0.1, epochs=2000).fit(x, y)
    preds = clf.predict(x)

    accuracy = (preds == y).mean()
    assert accuracy > 0.9
