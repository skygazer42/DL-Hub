import numpy as np
import pytest

from ml_algorithms.python.nmf import NMF


def test_nmf_fit_transform_reconstructs_low_rank_nonnegative_matrix() -> None:
    rng = np.random.default_rng(0)
    w_true = rng.random((60, 3))
    h_true = rng.random((3, 40))
    x = w_true @ h_true

    model = NMF(n_components=3, max_iter=800, tol=1e-7, random_state=0)
    w = model.fit_transform(x)

    assert w.shape == (x.shape[0], 3)
    assert model.basis_.shape == (x.shape[0], 3)
    assert model.components_.shape == (3, x.shape[1])
    assert isinstance(model.reconstruction_err_, float)

    x_hat = model.basis_ @ model.components_
    rel_err = np.linalg.norm(x - x_hat) / np.linalg.norm(x)
    assert rel_err < 1e-2

    assert np.all(model.basis_ >= 0.0)
    assert np.all(model.components_ >= 0.0)


def test_nmf_raises_on_negative_entries() -> None:
    x = np.array([[1.0, -0.1], [0.2, 0.3]], dtype=np.float64)
    with pytest.raises(ValueError):
        NMF(n_components=2, random_state=0).fit(x)


def test_nmf_is_deterministic_given_seed() -> None:
    rng = np.random.default_rng(123)
    x = rng.random((30, 10))

    model1 = NMF(n_components=4, max_iter=50, tol=0.0, random_state=1).fit(x)
    model2 = NMF(n_components=4, max_iter=50, tol=0.0, random_state=1).fit(x)

    assert np.allclose(model1.basis_, model2.basis_, atol=1e-12)
    assert np.allclose(model1.components_, model2.components_, atol=1e-12)
