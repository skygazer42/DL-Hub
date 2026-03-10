import pytest

torch = pytest.importorskip("torch")


def test_gnn_cora_gcn_data_shapes_smoke() -> None:
    from tracks.gnn.lesson_04_cora_node_classification_gcn.data import load_cora

    data = load_cora()
    assert tuple(data.features.shape) == (2708, 1433)
    assert tuple(data.labels.shape) == (2708,)
    assert data.adj.is_sparse
    assert data.adj.shape == (2708, 2708)
    assert data.adj_row.is_sparse
    assert data.adj_row.shape == (2708, 2708)

    assert len(data.idx_train) == 140
    assert len(data.idx_val) == 300
    assert len(data.idx_test) == 1000


def test_gnn_cora_gcn_forward_shape_smoke() -> None:
    from tracks.gnn.lesson_04_cora_node_classification_gcn.data import load_cora
    from tracks.gnn.lesson_04_cora_node_classification_gcn.model import GCN, ModelConfig

    data = load_cora()
    model = GCN(
        ModelConfig(in_features=1433, hidden_features=16, num_classes=int(data.labels.max()) + 1)
    )
    logits = model(data.features, data.adj)
    assert tuple(logits.shape) == (2708, int(data.labels.max()) + 1)
