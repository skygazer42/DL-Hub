import pytest


torch = pytest.importorskip("torch")


def test_gnn_karate_dataset_shapes_smoke() -> None:
    from tracks.gnn.datasets.karate import load_karate

    data = load_karate()
    assert data.num_nodes == 34
    assert data.edge_index.shape[0] == 2
    assert data.adj.shape == (34, 34)
    assert torch.allclose(torch.diag(data.adj), torch.ones(34))


def test_gnn_sdne_forward_and_loss_smoke() -> None:
    from tracks.gnn.datasets.karate import load_karate
    from tracks.gnn.lesson_07_sdne_karate_embedding.model import ModelConfig, SDNE, sdne_loss

    graph = load_karate()
    adj = graph.adj
    model = SDNE(ModelConfig(num_nodes=graph.num_nodes, embed_dim=8, hidden_dim=32, dropout=0.0))
    recon_logits, z = model(adj)
    assert recon_logits.shape == (34, 34)
    assert z.shape == (34, 8)

    loss, recon, smooth = sdne_loss(
        recon_logits=recon_logits,
        adj=adj,
        embeddings=z,
        edge_index=graph.edge_index,
        lambda_smooth=1.0,
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(recon)
    assert torch.isfinite(smooth)

