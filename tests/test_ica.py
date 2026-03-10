import itertools

import numpy as np

from ml_algorithms.python.ica import FastICA


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return x / (std + 1e-12)


def _abs_corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _standardize(a)
    b = _standardize(b)
    return np.abs((a.T @ b) / float(a.shape[0]))


def _best_permutation(corr: np.ndarray) -> tuple[int, ...]:
    n = int(corr.shape[0])
    best_score = -1.0
    best_perm: tuple[int, ...] | None = None
    for perm in itertools.permutations(range(n)):
        score = float(sum(corr[i, perm[i]] for i in range(n)))
        if score > best_score:
            best_score = score
            best_perm = perm
    assert best_perm is not None
    return best_perm


def test_fastica_recovers_sources_up_to_permutation_and_sign() -> None:
    rng = np.random.default_rng(0)
    n_samples = 2000
    n_components = 3

    s = rng.laplace(size=(n_samples, n_components))

    mixing = rng.normal(size=(n_components, n_components))
    while np.linalg.matrix_rank(mixing) < n_components:
        mixing = rng.normal(size=(n_components, n_components))

    x = s @ mixing.T

    model = FastICA(n_components=n_components, max_iter=400, tol=1e-5, random_state=0).fit(x)
    recovered = model.transform(x)

    assert model.components_.shape == (n_components, x.shape[1])
    assert model.mixing_.shape == (x.shape[1], n_components)
    assert model.mean_.shape == (x.shape[1],)

    corr = _abs_corr_matrix(s, recovered)
    perm = _best_permutation(corr)
    matched = np.array([corr[i, perm[i]] for i in range(n_components)], dtype=np.float64)
    assert np.all(matched > 0.9)


def test_fastica_is_deterministic_given_seed() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(800, 3))

    model1 = FastICA(n_components=2, max_iter=300, tol=1e-5, random_state=7).fit(x)
    model2 = FastICA(n_components=2, max_iter=300, tol=1e-5, random_state=7).fit(x)

    assert np.allclose(model1.mean_, model2.mean_)
    assert np.allclose(model1.components_, model2.components_)
    assert np.allclose(model1.mixing_, model2.mixing_)
    assert np.allclose(model1.transform(x), model2.transform(x))
