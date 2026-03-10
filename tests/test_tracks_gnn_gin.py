import pytest

torch = pytest.importorskip("torch")


def test_gnn_lesson_02_gin_shapes_smoke() -> None:
    from tracks.gnn.lesson_02_gin_toy_graph_classification.data import DataConfig, get_dataloaders
    from tracks.gnn.lesson_02_gin_toy_graph_classification.model import (
        GINGraphClassifier,
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

    model = GINGraphClassifier(
        ModelConfig(in_features=2, hidden_features=16, num_layers=3, num_classes=2)
    )
    logits = model((x, adj))
    assert tuple(logits.shape) == (4, 2)
