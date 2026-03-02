import pytest


torch = pytest.importorskip("torch")


def test_pointcloud_lesson_01_dataloaders_and_forward_smoke() -> None:
    from tracks.pointcloud.lesson_01_pointnet_toy_classification.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_01_pointnet_toy_classification.model import ModelConfig, PointNetClassifier

    train_loader, _ = get_dataloaders(
        DataConfig(num_samples=32, num_points=64, batch_size=8, val_fraction=0.2, seed=0, num_workers=0)
    )
    points, labels = next(iter(train_loader))
    assert points.shape == (8, 64, 3)
    assert labels.shape == (8,)

    model = PointNetClassifier(ModelConfig(hidden_features=32, num_classes=2, dropout=0.0))
    logits = model(points)
    assert logits.shape == (8, 2)


def test_pointcloud_lesson_02_dgcnn_forward_smoke() -> None:
    from tracks.pointcloud.lesson_02_dgcnn_toy_classification.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_02_dgcnn_toy_classification.model import DGCNNClassifier, ModelConfig

    train_loader, _ = get_dataloaders(
        DataConfig(num_samples=32, num_points=32, batch_size=4, val_fraction=0.2, seed=0, num_workers=0)
    )
    points, labels = next(iter(train_loader))
    assert points.shape == (4, 32, 3)
    assert labels.shape == (4,)

    model = DGCNNClassifier(ModelConfig(k=5, hidden_features=16, dropout=0.0, num_classes=2, dynamic_graph=True))
    logits = model(points)
    assert logits.shape == (4, 2)


def test_pointcloud_lesson_03_pointnet2_forward_smoke() -> None:
    from tracks.pointcloud.lesson_03_pointnet2_toy_classification.data import DataConfig, get_dataloaders
    from tracks.pointcloud.lesson_03_pointnet2_toy_classification.model import ModelConfig, PointNet2Classifier

    train_loader, _ = get_dataloaders(
        DataConfig(num_samples=32, num_points=64, batch_size=4, val_fraction=0.2, seed=0, num_workers=0)
    )
    points, labels = next(iter(train_loader))
    assert points.shape == (4, 64, 3)
    assert labels.shape == (4,)

    model = PointNet2Classifier(
        ModelConfig(
            npoint1=16,
            k1=8,
            npoint2=4,
            k2=4,
            hidden_features=32,
            dropout=0.0,
            num_classes=2,
        )
    )
    logits = model(points)
    assert logits.shape == (4, 2)
