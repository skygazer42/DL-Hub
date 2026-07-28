import pytest

torch = pytest.importorskip("torch")


def test_vision_compact_keypoint_regression_shapes_and_loss_smoke() -> None:
    from tracks.vision.lesson_07_compact_keypoint_regression.data import DataConfig, get_dataloaders
    from tracks.vision.lesson_07_compact_keypoint_regression.model import KeypointRegressor, ModelConfig

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=64, batch_size=8, image_size=64, val_fraction=0.2, seed=0, num_workers=0
        )
    )
    x, target = next(iter(train_loader))

    assert tuple(x.shape) == (8, 1, 64, 64)
    assert tuple(target.shape) == (8, 2)
    assert target.dtype == torch.float32

    model = KeypointRegressor(
        ModelConfig(in_channels=1, hidden_channels=16, num_blocks=2, dropout=0.0)
    )
    pred = model(x)
    assert tuple(pred.shape) == (8, 2)

    loss = torch.nn.MSELoss()(pred, target)
    assert torch.isfinite(loss)
    loss.backward()
