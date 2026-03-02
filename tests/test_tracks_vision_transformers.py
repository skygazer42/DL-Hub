import pytest


torch = pytest.importorskip("torch")


def test_vision_vit_toy_quadrant_forward_and_loss_smoke() -> None:
    from tracks.vision.toy_shapes import DataConfig, get_dataloaders
    from tracks.vision.lesson_05_vit_toy_classification.model import ModelConfig, ViTClassifier

    train_loader, _ = get_dataloaders(
        DataConfig(num_samples=64, batch_size=8, image_size=64, val_fraction=0.2, seed=0, num_workers=0)
    )
    x, y = next(iter(train_loader))

    model = ViTClassifier(
        ModelConfig(
            image_size=64,
            patch_size=8,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            ff_dim=128,
            dropout=0.0,
            num_classes=4,
        )
    )
    logits = model(x)
    assert tuple(logits.shape) == (8, 4)

    loss = torch.nn.CrossEntropyLoss()(logits, y)
    assert torch.isfinite(loss)
    loss.backward()


def test_vision_swin_toy_quadrant_forward_and_loss_smoke() -> None:
    from tracks.vision.toy_shapes import DataConfig, get_dataloaders
    from tracks.vision.lesson_06_swin_toy_classification.model import ModelConfig, SwinTinyClassifier

    train_loader, _ = get_dataloaders(
        DataConfig(num_samples=64, batch_size=8, image_size=64, val_fraction=0.2, seed=0, num_workers=0)
    )
    x, y = next(iter(train_loader))

    model = SwinTinyClassifier(
        ModelConfig(
            image_size=64,
            patch_size=4,
            embed_dim=64,
            num_heads=4,
            depth=2,
            window_size=4,
            mlp_ratio=2.0,
            dropout=0.0,
            num_classes=4,
        )
    )
    logits = model(x)
    assert tuple(logits.shape) == (8, 4)

    loss = torch.nn.CrossEntropyLoss()(logits, y)
    assert torch.isfinite(loss)
    loss.backward()

