import pytest


torch = pytest.importorskip("torch")


def test_vision_synth_seg_unet_shapes_and_one_step_smoke() -> None:
    from dlhub.training.loop import evaluate_binary_segmentation, fit_binary_segmentation
    from tracks.vision.lesson_08_synthetic_segmentation_unet.data import DataConfig, get_dataloaders
    from tracks.vision.lesson_08_synthetic_segmentation_unet.model import ModelConfig, TinyUNet

    train_loader, val_loader = get_dataloaders(
        DataConfig(
            num_samples=64,
            batch_size=4,
            image_size=64,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            noise_std=0.1,
            min_rect=10,
            max_rect=20,
        )
    )
    x, y = next(iter(train_loader))

    assert tuple(x.shape) == (4, 1, 64, 64)
    assert tuple(y.shape) == (4, 1, 64, 64)

    model = TinyUNet(ModelConfig(in_channels=1, base_channels=16, dropout=0.0))
    logits = model(x)
    assert tuple(logits.shape) == (4, 1, 64, 64)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_stats = fit_binary_segmentation(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        max_batches=1,
        threshold=0.5,
    )
    eval_stats = evaluate_binary_segmentation(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=torch.device("cpu"),
        max_batches=1,
        threshold=0.5,
    )
    assert torch.isfinite(torch.tensor(train_stats.loss))
    assert 0.0 <= train_stats.iou <= 1.0
    assert torch.isfinite(torch.tensor(eval_stats.loss))
    assert 0.0 <= eval_stats.iou <= 1.0


def test_vision_synth_seg_torchvision_model_forward_loss_backward_smoke() -> None:
    pytest.importorskip("torchvision")

    from tracks.vision.lesson_08_synthetic_segmentation_unet.data import DataConfig, get_dataloaders
    from tracks.vision.lesson_08_synthetic_segmentation_unet.model import ModelConfig, build_model

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=16,
            batch_size=2,
            image_size=64,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            noise_std=0.1,
            min_rect=10,
            max_rect=20,
        )
    )
    x, y = next(iter(train_loader))

    model = build_model(ModelConfig(arch="tvseg:lraspp_mobilenet_v3_large", in_channels=1, base_channels=16, dropout=0.0))
    logits = model(x)
    assert tuple(logits.shape) == (2, 1, 64, 64)

    loss = torch.nn.BCEWithLogitsLoss()(logits, y)
    assert torch.isfinite(loss)
    loss.backward()
