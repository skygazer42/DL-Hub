import numpy as np

from ml_algorithms.python.gradient_boosting import GradientBoostingRegressor


def test_gradient_boosting_regressor_beats_mean_baseline_on_nonlinear_signal() -> None:
    rng = np.random.default_rng(3)
    x = np.linspace(-3.0, 3.0, 240).reshape(-1, 1)
    noise = rng.normal(scale=0.08, size=x.shape[0])
    y = np.sin(1.7 * x[:, 0]) + 0.15 * x[:, 0] + noise

    baseline = np.full_like(y, fill_value=float(np.mean(y)))
    baseline_mse = np.mean((baseline - y) ** 2)

    model = GradientBoostingRegressor(
        n_estimators=30,
        learning_rate=0.15,
        max_depth=2,
        min_samples_split=4,
    ).fit(x, y)
    preds = model.predict(x)

    mse = np.mean((preds - y) ** 2)
    assert mse < baseline_mse * 0.35
