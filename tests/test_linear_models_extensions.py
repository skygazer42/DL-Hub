import numpy as np

from ml_algorithms.python.linear_models import RidgeRegression, SoftmaxRegression


def test_ridge_regression_shrinks_weights_while_keeping_low_error() -> None:
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(320, 5))
    raw -= raw.mean(axis=0, keepdims=True)
    x, _ = np.linalg.qr(raw)

    true_w = np.array([1.5, -2.0, 0.75, 0.0, 0.5], dtype=np.float64)
    y = x @ true_w + rng.normal(scale=0.01, size=x.shape[0])

    ols = RidgeRegression(alpha=0.0).fit(x, y)
    ridge = RidgeRegression(alpha=1.0).fit(x, y)

    ols_norm = float(np.linalg.norm(ols.weights_[1:]))
    ridge_norm = float(np.linalg.norm(ridge.weights_[1:]))
    assert ridge_norm < ols_norm

    preds = ridge.predict(x)
    mse = float(np.mean((preds - y) ** 2))
    assert mse < 1e-2


def test_softmax_regression_fits_three_separable_classes() -> None:
    rng = np.random.default_rng(0)
    centers = np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.5]], dtype=np.float64)
    x = np.vstack([rng.normal(loc=center, scale=0.4, size=(120, 2)) for center in centers])
    y = np.array([0] * 120 + [1] * 120 + [2] * 120, dtype=int)

    clf = SoftmaxRegression(learning_rate=0.2, epochs=2500).fit(x, y)
    preds = clf.predict(x)

    accuracy = float((preds == y).mean())
    assert accuracy > 0.9

    proba = clf.predict_proba(x[:10])
    assert proba.shape == (10, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
