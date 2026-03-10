import numpy as np
import pytest

from ml_algorithms.python.spectral_clustering import SpectralClustering


def _concentric_circles(
    rng: np.random.Generator,
    *,
    n_inner: int,
    n_outer: int,
    noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    angles_inner = rng.uniform(0.0, 2.0 * np.pi, size=n_inner)
    inner = np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    inner = inner + rng.normal(scale=noise, size=inner.shape)

    angles_outer = rng.uniform(0.0, 2.0 * np.pi, size=n_outer)
    outer = 3.0 * np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    outer = outer + rng.normal(scale=noise, size=outer.shape)

    x = np.vstack([inner, outer]).astype(np.float64, copy=False)
    y = np.array([0] * n_inner + [1] * n_outer, dtype=int)
    return x, y


def test_spectral_clustering_separates_concentric_circles() -> None:
    rng = np.random.default_rng(0)
    x, _ = _concentric_circles(rng, n_inner=120, n_outer=120, noise=0.04)

    model = SpectralClustering(n_clusters=2, gamma=1.0, random_state=0).fit(x)
    labels = model.labels_

    assert labels.shape == (x.shape[0],)
    assert np.unique(labels).size == 2

    radii = np.linalg.norm(x, axis=1)
    means = [float(radii[labels == k].mean()) for k in np.unique(labels)]
    means_sorted = np.sort(means)
    assert means_sorted[0] < 1.5
    assert means_sorted[1] > 2.5


def test_spectral_clustering_is_deterministic_given_seed() -> None:
    rng = np.random.default_rng(123)
    x, _ = _concentric_circles(rng, n_inner=80, n_outer=80, noise=0.05)

    model1 = SpectralClustering(n_clusters=2, gamma=1.0, random_state=1).fit(x)
    model2 = SpectralClustering(n_clusters=2, gamma=1.0, random_state=1).fit(x)
    assert np.array_equal(model1.labels_, model2.labels_)


def test_spectral_clustering_validates_inputs() -> None:
    x = np.zeros((5, 2), dtype=np.float64)

    with pytest.raises(ValueError):
        SpectralClustering(n_clusters=1, gamma=1.0).fit(x)
    with pytest.raises(ValueError):
        SpectralClustering(n_clusters=10, gamma=1.0).fit(x)
    with pytest.raises(ValueError):
        SpectralClustering(n_clusters=2, gamma=0.0).fit(x)
