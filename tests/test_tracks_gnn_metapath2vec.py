import pytest

torch = pytest.importorskip("torch")


def test_gnn_metapath2vec_loss_smoke() -> None:
    from tracks.gnn.lesson_09_metapath2vec_toy_hetero_embedding.data import (
        DataConfig,
        build_training_pairs,
    )
    from tracks.gnn.lesson_09_metapath2vec_toy_hetero_embedding.model import (
        MetaPath2Vec,
        ModelConfig,
    )

    graph, pairs, sampler = build_training_pairs(
        DataConfig(num_walks=30, walk_length=8, window_size=2, care_type=1, seed=0)
    )
    model = MetaPath2Vec(ModelConfig(num_nodes=graph.num_nodes, embed_dim=16, sparse=False))

    center = pairs.centers[:32]
    context = pairs.contexts[:32]
    neg = sampler.sample(context, k=3)

    loss = model.loss(center=center, context=context, neg_context=neg)
    assert torch.isfinite(loss)
