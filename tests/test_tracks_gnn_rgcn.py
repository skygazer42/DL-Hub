import pytest

torch = pytest.importorskip("torch")


def test_gnn_rgcn_forward_smoke() -> None:
    from tracks.gnn.lesson_11_rgcn_toy_node_classification.data import (
        DataConfig,
        load_toy_rel_graph,
    )
    from tracks.gnn.lesson_11_rgcn_toy_node_classification.model import RGCN, ModelConfig

    data = load_toy_rel_graph(
        DataConfig(
            num_nodes=60,
            num_rels=3,
            num_classes=3,
            feature_dim=8,
            edges_per_node=2,
            seed=0,
        )
    )
    model = RGCN(
        ModelConfig(
            in_features=int(data.features.shape[1]),
            hidden_features=16,
            num_classes=int(data.num_classes),
            num_rels=int(data.num_rels),
            num_bases=2,
            dropout=0.0,
        )
    )

    logits = model(
        data.features,
        edge_index=data.edge_index,
        edge_type=data.edge_type,
        edge_norm=data.edge_norm,
    )
    assert tuple(logits.shape) == (int(data.num_nodes), int(data.num_classes))

    loss = torch.nn.functional.cross_entropy(logits[data.idx_train], data.labels[data.idx_train])
    assert torch.isfinite(loss)
