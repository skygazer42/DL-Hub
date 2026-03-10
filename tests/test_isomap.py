import numpy as np
import pytest

from ml_algorithms.python.isomap import Isomap


def test_isomap_unrolls_semicircle_to_one_dimension() -> None:
    rng = np.random.default_rng(0)
    n_samples = 220
    theta = np.linspace(0.0, np.pi, n_samples, dtype=np.float64)

    x = np.column_stack([np.cos(theta), np.sin(theta)])
    x = x + rng.normal(scale=0.01, size=x.shape)

    model = Isomap(n_neighbors=10, n_components=1).fit(x)
    embedding = model.embedding_

    assert embedding.shape == (n_samples, 1)

    corr = np.corrcoef(embedding[:, 0], theta)[0, 1]
    assert abs(float(corr)) > 0.95


def test_isomap_is_deterministic() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(120, 3))

    model1 = Isomap(n_neighbors=12, n_components=2).fit(x)
    model2 = Isomap(n_neighbors=12, n_components=2).fit(x)
    assert np.allclose(model1.embedding_, model2.embedding_, atol=1e-12)


def test_isomap_raises_for_disconnected_graph() -> None:
    x = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 0.0],
            [10.1, 0.0],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ValueError):
        Isomap(n_neighbors=1, n_components=2).fit(x)


def test_isomap_validates_inputs() -> None:
    x = np.zeros((5, 2), dtype=np.float64)

    with pytest.raises(ValueError):
        Isomap(n_neighbors=0, n_components=1).fit(x)
    with pytest.raises(ValueError):
        Isomap(n_neighbors=5, n_components=1).fit(x)
    with pytest.raises(ValueError):
        Isomap(n_neighbors=2, n_components=0).fit(x)
    with pytest.raises(ValueError):
        Isomap(n_neighbors=2, n_components=10).fit(x)
