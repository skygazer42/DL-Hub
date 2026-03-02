import pytest


torch = pytest.importorskip("torch")


def test_gnn_pinsage_encode_and_loss_smoke() -> None:
    from tracks.gnn.lesson_10_pinsage_toy_recommender.data import DataConfig, build_toy_recommender_data
    from tracks.gnn.lesson_10_pinsage_toy_recommender.model import ModelConfig, PinSAGEItemEncoder

    data = build_toy_recommender_data(
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
    neg_repr = model.encode(item_ids=neg_flat, neighbors=data.item_neighbors[neg_flat]).view(8, 3, -1)

    loss = model.loss(center=center, pos=pos, neg=neg_repr)
    assert torch.isfinite(loss)

