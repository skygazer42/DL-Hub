import pytest

torch = pytest.importorskip("torch")


def _accuracy(probs: "torch.Tensor", labels: "torch.Tensor", idx: "torch.Tensor") -> float:
    pred = probs[idx].argmax(dim=1)
    return float((pred == labels[idx]).float().mean().item())


def test_gnn_cora_adj_row_is_row_normalized() -> None:
    from tracks.gnn.datasets.cora import load_cora

    data = load_cora()
    row_sums = torch.sparse.sum(data.adj_row, dim=1).to_dense()
    assert float(row_sums.min().item()) > 0.999
    assert float(row_sums.max().item()) < 1.001


def test_gnn_cora_label_propagation_smoke() -> None:
    from tracks.gnn.datasets.cora import load_cora
    from tracks.gnn.lesson_05_label_propagation_cora.model import (
        LabelPropagation,
        LabelPropagationConfig,
    )

    data = load_cora()
    model = LabelPropagation(LabelPropagationConfig(num_layers=3, alpha=0.9, clamp_labeled=True))
    probs = model(adj_row=data.adj_row, labels=data.labels, idx_labeled=data.idx_train)

    assert probs.shape == (2708, int(data.labels.max().item()) + 1)
    assert torch.isfinite(probs).all()
    assert _accuracy(probs, data.labels, data.idx_train) == 1.0


def test_gnn_cora_graphsage_forward_shape_smoke() -> None:
    from tracks.gnn.datasets.cora import load_cora
    from tracks.gnn.lesson_06_graphsage_cora.model import GraphSAGE, ModelConfig

    data = load_cora()
    model = GraphSAGE(
        ModelConfig(
            in_features=int(data.features.shape[1]),
            hidden_features=16,
            num_classes=int(data.labels.max().item()) + 1,
            dropout=0.0,
        )
    )
    logits = model(data.features, data.adj_row)
    assert logits.shape == (2708, int(data.labels.max().item()) + 1)
