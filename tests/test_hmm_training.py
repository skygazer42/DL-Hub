import numpy as np

from ml_algorithms.python.hmm import CategoricalHMM


def _sample_hmm(
    rng: np.random.Generator,
    startprob: np.ndarray,
    transmat: np.ndarray,
    emissionprob: np.ndarray,
    length: int,
) -> np.ndarray:
    startprob = np.asarray(startprob, dtype=np.float64).ravel()
    transmat = np.asarray(transmat, dtype=np.float64)
    emissionprob = np.asarray(emissionprob, dtype=np.float64)

    n_states = int(startprob.size)
    n_observations = int(emissionprob.shape[1])

    states = np.empty((length,), dtype=int)
    observations = np.empty((length,), dtype=int)

    states[0] = int(rng.choice(n_states, p=startprob))
    observations[0] = int(rng.choice(n_observations, p=emissionprob[states[0]]))
    for t in range(1, int(length)):
        states[t] = int(rng.choice(n_states, p=transmat[states[t - 1]]))
        observations[t] = int(rng.choice(n_observations, p=emissionprob[states[t]]))
    return observations


def test_hmm_fit_with_observations_increases_log_likelihood() -> None:
    rng = np.random.default_rng(0)

    start_true = np.array([0.75, 0.25], dtype=np.float64)
    trans_true = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64)
    emit_true = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=np.float64)

    sequences = [_sample_hmm(rng, start_true, trans_true, emit_true, length=80) for _ in range(6)]

    start_init = np.array([0.55, 0.45], dtype=np.float64)
    trans_init = np.array([[0.6, 0.4], [0.4, 0.6]], dtype=np.float64)
    emit_init = np.array([[0.34, 0.33, 0.33], [0.33, 0.34, 0.33]], dtype=np.float64)

    model = CategoricalHMM(startprob=start_init, transmat=trans_init, emissionprob=emit_init).fit()

    ll_before = sum(model.score(seq) for seq in sequences)

    model.fit(sequences, n_iter=25, tol=1e-6)
    ll_after = sum(model.score(seq) for seq in sequences)

    assert ll_after > ll_before + 1e-3
    assert np.allclose(model.startprob_.sum(), 1.0)
    assert np.allclose(model.transmat_.sum(axis=1), 1.0)
    assert np.allclose(model.emissionprob_.sum(axis=1), 1.0)
