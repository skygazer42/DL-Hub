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

