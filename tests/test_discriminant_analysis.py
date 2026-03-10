import numpy as np

from ml_algorithms.python.discriminant_analysis import LDAClassifier, QDAClassifier


def test_lda_classifier_separates_three_gaussians() -> None:
    rng = np.random.default_rng(0)
    n_per_class = 250
    cov = np.array([[0.7, 0.0], [0.0, 0.7]], dtype=np.float64)

    x0 = rng.multivariate_normal(mean=(-3.0, 0.0), cov=cov, size=n_per_class)
    x1 = rng.multivariate_normal(mean=(0.0, 3.0), cov=cov, size=n_per_class)
    x2 = rng.multivariate_normal(mean=(3.0, 0.0), cov=cov, size=n_per_class)

    x = np.vstack([x0, x1, x2])
    y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

    clf = LDAClassifier(reg=1e-6).fit(x, y)
    preds = clf.predict(x)

    accuracy = float(np.mean(preds == y))
    assert accuracy > 0.9


def test_qda_classifier_handles_singular_covariance_with_regularization() -> None:
    rng = np.random.default_rng(1)
    n_per_class = 500

    # Construct a dataset with perfectly collinear features (singular covariance).
    # Regularization should make the per-class covariance matrices invertible.
    x1_0 = rng.normal(loc=-1.5, scale=1.0, size=n_per_class)
    x1_1 = rng.normal(loc=1.5, scale=1.0, size=n_per_class)
    x0 = np.column_stack([x1_0, 2.0 * x1_0])
    x1 = np.column_stack([x1_1, 2.0 * x1_1])

    x = np.vstack([x0, x1])
    y = np.array([0] * n_per_class + [1] * n_per_class)

    clf = QDAClassifier(reg=1e-2).fit(x, y)
    preds = clf.predict(x)

    accuracy = float(np.mean(preds == y))
    assert accuracy > 0.9
