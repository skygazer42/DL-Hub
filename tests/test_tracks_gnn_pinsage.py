import pytest

torch = pytest.importorskip("torch")


def test_gnn_pinsage_encode_and_loss_smoke() -> None:
    from tracks.gnn.lesson_10_pinsage_compact_recommender.data import (
        DataConfig,
        build_baseline_recommender_data,
    )
    from tracks.gnn.lesson_10_pinsage_compact_recommender.model import (
        ModelConfig,
        PinSAGEItemEncoder,
    )

    data = build_baseline_recommender_data(
        DataConfig(
            num_users=32,
            num_items=64,
            interactions_per_user=8,
            test_fraction=0.25,
            num_random_walks=8,
            num_neighbors=4,
            seed=0,
        )
    )
    model = PinSAGEItemEncoder(
        ModelConfig(num_items=data.num_items, embed_dim=16, num_neighbors=4, normalize=True)
    )

    item_ids = torch.arange(0, 8, dtype=torch.long)
    neigh = data.item_neighbors[item_ids]
    center = model.encode(item_ids=item_ids, neighbors=neigh)
    assert tuple(center.shape) == (8, 16)

    pos_ids = torch.where(neigh[:, 0] >= 0, neigh[:, 0], item_ids)
    pos = model.encode(item_ids=pos_ids, neighbors=data.item_neighbors[pos_ids])

    neg_ids = torch.randint(low=0, high=data.num_items, size=(8, 3), dtype=torch.long)
    neg_flat = neg_ids.reshape(-1)
    neg_repr = model.encode(item_ids=neg_flat, neighbors=data.item_neighbors[neg_flat]).view(
        8, 3, -1
    )

    loss = model.loss(center=center, pos=pos, neg=neg_repr)
    assert torch.isfinite(loss)


def test_gnn_pinsage_sampling_preserves_cpu_generator_sequence_on_target_device() -> None:
    from tracks.gnn.lesson_10_pinsage_compact_recommender.train import _sample_item_ids

    expected_gen = torch.Generator().manual_seed(123)
    expected = torch.randint(0, 17, (5, 3), generator=expected_gen)
    actual_gen = torch.Generator().manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actual = _sample_item_ids(high=17, size=(5, 3), gen=actual_gen, device=device)

    assert actual.device.type == device.type
    assert torch.equal(actual.cpu(), expected)
