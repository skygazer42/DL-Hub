import numpy as np
import pytest

from dlhub.data.splits import train_val_split_indices


def test_train_val_split_indices_is_deterministic_and_disjoint() -> None:
    train_a, val_a = train_val_split_indices(n=100, val_fraction=0.2, seed=0)
    train_b, val_b = train_val_split_indices(n=100, val_fraction=0.2, seed=0)

    assert train_a == train_b
    assert val_a == val_b

    assert len(train_a) == 80
    assert len(val_a) == 20

    assert set(train_a).isdisjoint(val_a)
    assert sorted(train_a + val_a) == list(range(100))


@pytest.mark.parametrize("n", [0, 1, -1])
def test_train_val_split_requires_enough_samples_for_both_partitions(n: int) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        train_val_split_indices(n=n)


@pytest.mark.parametrize("n", [True, 3.5])
def test_train_val_split_rejects_non_integer_dataset_sizes(n: object) -> None:
    with pytest.raises(TypeError, match="n must be an integer"):
        train_val_split_indices(n=n)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [True, 1.5])
def test_train_val_split_rejects_ambiguous_seeds(seed: object) -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        train_val_split_indices(n=10, seed=seed)  # type: ignore[arg-type]


def test_train_val_split_accepts_numpy_integer_seed_and_keeps_both_sides_nonempty() -> None:
    train, val = train_val_split_indices(n=np.int64(2), seed=np.int64(7))

    assert len(train) == len(val) == 1
