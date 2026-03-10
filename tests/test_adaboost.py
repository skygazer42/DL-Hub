import numpy as np

from ml_algorithms.python.adaboost import AdaBoostClassifier


def test_adaboost_fits_simple_or_rule_and_predict_proba() -> None:
    rng = np.random.default_rng(0)
    x = rng.random((600, 2))
    y = ((x[:, 0] > 0.7) | (x[:, 1] > 0.7)).astype(int)

    model = AdaBoostClassifier(n_estimators=10, learning_rate=1.0, random_state=0).fit(x, y)
    preds = model.predict(x)

    assert (preds == y).mean() > 0.98

    proba = model.predict_proba(x[:10])
    assert proba.shape == (10, 2)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_adaboost_is_deterministic_given_seed() -> None:
    rng = np.random.default_rng(123)
    x = rng.random((200, 2))
    y = ((x[:, 0] > 0.5) | (x[:, 1] > 0.5)).astype(int)

    model1 = AdaBoostClassifier(n_estimators=5, learning_rate=1.0, random_state=1).fit(x, y)
    model2 = AdaBoostClassifier(n_estimators=5, learning_rate=1.0, random_state=1).fit(x, y)

    assert np.array_equal(model1.predict(x), model2.predict(x))
