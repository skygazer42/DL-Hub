import pytest

torch = pytest.importorskip("torch")


def test_gnn_lesson_03_gat_shapes_smoke() -> None:
    from tracks.gnn.lesson_03_gat_compact_graph_classification.data import DataConfig, get_dataloaders
    from tracks.gnn.lesson_03_gat_compact_graph_classification.model import (
        GATGraphClassifier,
        ModelConfig,
    )

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_graphs=32, num_nodes=10, batch_size=4, val_fraction=0.2, seed=0, num_workers=0
        )
    )
    (x, adj), y = next(iter(train_loader))

    assert tuple(x.shape) == (4, 10, 2)
    assert tuple(adj.shape) == (4, 10, 10)
    assert tuple(y.shape) == (4,)

    model = GATGraphClassifier(
        ModelConfig(in_features=2, hidden_features=16, num_heads=4, num_classes=2, dropout=0.1)
    )
    logits = model((x, adj))
    assert tuple(logits.shape) == (4, 2)
