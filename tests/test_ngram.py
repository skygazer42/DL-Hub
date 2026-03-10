import numpy as np

from ml_algorithms.python.ngram import NGramLanguageModel


def test_bigram_predicts_most_likely_next_token() -> None:
    sequences = [
        np.array([0, 1, 0, 1, 0, 1, 2], dtype=int),
        np.array([0, 1, 0, 1], dtype=int),
    ]

    model = NGramLanguageModel(n=2, alpha=1.0).fit(sequences)

    assert model.predict_next([0]) == 1
    assert model.predict_next([1]) == 0


def test_bigram_unseen_context_is_uniform_under_laplace_smoothing() -> None:
    sequences = [
        np.array([0, 1, 0, 1, 0, 1, 2], dtype=int),
        np.array([0, 1, 0, 1], dtype=int),
    ]

    model = NGramLanguageModel(n=2, alpha=1.0).fit(sequences)

    proba = model.predict_next_proba([2])
    assert proba.shape == (3,)
    assert np.allclose(proba.sum(), 1.0, atol=1e-12)
    assert np.allclose(proba, np.full((3,), 1.0 / 3.0), atol=1e-12)
