import itertools

import numpy as np

from ml_algorithms.python.hmm import CategoricalHMM


def _joint_probability(
    startprob: np.ndarray,
    transmat: np.ndarray,
    emissionprob: np.ndarray,
    states: tuple[int, ...],
    observations: np.ndarray,
) -> float:
    prob = float(startprob[states[0]] * emissionprob[states[0], observations[0]])
    for t in range(1, int(observations.size)):
        prob *= float(transmat[states[t - 1], states[t]] * emissionprob[states[t], observations[t]])
    return prob


def _bruteforce_log_likelihood_and_best_path(
    startprob: np.ndarray,
    transmat: np.ndarray,
    emissionprob: np.ndarray,
    observations: np.ndarray,
) -> tuple[float, np.ndarray]:
    observations = np.asarray(observations, dtype=int).ravel()
    n_states = int(startprob.size)
    t = int(observations.size)

    total = 0.0
    best_prob = -1.0
    best_states: tuple[int, ...] | None = None

    for states in itertools.product(range(n_states), repeat=t):
        prob = _joint_probability(startprob, transmat, emissionprob, states, observations)
        total += prob
        if prob > best_prob:
            best_prob = prob
            best_states = states

    assert total > 0.0
    assert best_states is not None
    return float(np.log(total)), np.asarray(best_states, dtype=int)


def test_hmm_score_matches_bruteforce_enumeration() -> None:
    startprob = np.array([0.6, 0.4], dtype=np.float64)
    transmat = np.array([[0.7, 0.3], [0.4, 0.6]], dtype=np.float64)
    emissionprob = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]], dtype=np.float64)

    observations = np.array([0, 1, 2, 1], dtype=int)
    expected_ll, _ = _bruteforce_log_likelihood_and_best_path(
        startprob, transmat, emissionprob, observations
    )

    model = CategoricalHMM(startprob=startprob, transmat=transmat, emissionprob=emissionprob).fit()
    ll = model.score(observations)
    assert np.allclose(ll, expected_ll, atol=1e-12)


def test_hmm_predict_viterbi_matches_best_joint_path() -> None:
    startprob = np.array([0.65, 0.35], dtype=np.float64)
    transmat = np.array([[0.85, 0.15], [0.2, 0.8]], dtype=np.float64)
    emissionprob = np.array([[0.05, 0.3, 0.65], [0.7, 0.25, 0.05]], dtype=np.float64)

    observations = np.array([2, 2, 1, 0, 0], dtype=int)
    _, best_states = _bruteforce_log_likelihood_and_best_path(
        startprob, transmat, emissionprob, observations
    )

    model = CategoricalHMM(startprob=startprob, transmat=transmat, emissionprob=emissionprob).fit()
    states = model.predict(observations)
    assert np.array_equal(states, best_states)
