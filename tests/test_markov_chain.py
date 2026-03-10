import numpy as np

from ml_algorithms.python.markov_chain import MarkovChain


def test_markov_chain_learns_deterministic_alternation() -> None:
    seq = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)

    chain = MarkovChain().fit([seq])

    assert chain.predict_next(0) == 1
    assert chain.predict_next(1) == 0

    state_to_index = {int(s): int(i) for i, s in enumerate(chain.states_)}
    i0 = state_to_index[0]
    i1 = state_to_index[1]

    assert np.allclose(chain.transition_matrix_[i0], np.array([0.0, 1.0]))
    assert np.allclose(chain.transition_matrix_[i1], np.array([1.0, 0.0]))


def test_markov_chain_estimates_transition_probabilities() -> None:
    seq1 = np.array([0, 0, 1, 0], dtype=int)
    seq2 = np.array([0, 1, 1, 1], dtype=int)

    chain = MarkovChain().fit([seq1, seq2])

    # Counts:
    # 0 -> 0 (1), 0 -> 1 (2)  => P(1|0)=2/3
    # 1 -> 0 (1), 1 -> 1 (2)  => P(1|1)=2/3
    state_to_index = {int(s): int(i) for i, s in enumerate(chain.states_)}
    i0 = state_to_index[0]
    i1 = state_to_index[1]

    assert chain.predict_next(0) == 1
    assert chain.predict_next(1) == 1

    assert np.allclose(chain.transition_matrix_[i0, i1], 2.0 / 3.0)
    assert np.allclose(chain.transition_matrix_[i1, i1], 2.0 / 3.0)
