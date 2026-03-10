import pytest

torch = pytest.importorskip("torch")


def test_gnn_line_loss_smoke() -> None:
    from tracks.gnn.datasets.karate import load_karate
    from tracks.gnn.lesson_08_line_karate_embedding.model import LINE, ModelConfig

    graph = load_karate(add_self_loops=False)
    model = LINE(ModelConfig(num_nodes=graph.num_nodes, embed_dim=8, order=2))

    # Sample a few directed edges from the dataset.
    src = graph.edge_index[0][:4]
    dst = graph.edge_index[1][:4]
    neg = torch.randint(low=0, high=graph.num_nodes, size=(4, 3), dtype=torch.long)

    loss = model.loss(src=src, dst=dst, neg_dst=neg)
    assert torch.isfinite(loss)
