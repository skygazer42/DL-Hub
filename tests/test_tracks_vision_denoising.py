import pytest


torch = pytest.importorskip("torch")


def test_vision_denoising_supervised_forward_loss_backward_smoke() -> None:
    from tracks.vision.lesson_10_synthetic_denoising.data import DataConfig, get_dataloaders
    from tracks.vision.lesson_10_synthetic_denoising.model import ModelConfig, build_model

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=64,
            batch_size=4,
            image_size=32,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            in_channels=1,
            noise_std=0.15,
            min_square=6,
            max_square=10,
            train_mode="supervised",
        )
    )
    noisy, clean = next(iter(train_loader))
    assert tuple(noisy.shape) == (4, 1, 32, 32)
    assert tuple(clean.shape) == (4, 1, 32, 32)

    for arch in [
        "dncnn:dncnn_tiny",
        "restormer:restormer_tiny",
        "nafnet:nafnet_tiny",
        "swinir:swinir_tiny",
        "ridnet:ridnet_tiny",
        "ffdnet:ffdnet_tiny",
        "drunet:drunet_tiny",
        "noise2noise_unet:n2n_unet_tiny",
        "ddpm_unet:ddpm_unet_tiny",
        "mirnet:mirnet_tiny",
        "mprnet:mprnet_tiny",
        "uformer:uformer_tiny",
    ]:
        model = build_model(ModelConfig(arch=arch, variant="", in_channels=1, sigma=0.15))
        pred = model(noisy)
        assert tuple(pred.shape) == (4, 1, 32, 32)
        loss = torch.nn.MSELoss()(pred, clean)
        assert torch.isfinite(loss)
        loss.backward()


def test_vision_denoising_noise2noise_training_pair_smoke() -> None:
    from tracks.vision.lesson_10_synthetic_denoising.data import DataConfig, get_dataloaders
    from tracks.vision.lesson_10_synthetic_denoising.model import ModelConfig, build_model

    train_loader, val_loader = get_dataloaders(
        DataConfig(
            num_samples=64,
            batch_size=4,
            image_size=32,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            in_channels=1,
            noise_std=0.15,
            min_square=6,
            max_square=10,
            train_mode="noise2noise",
        )
    )
    noisy1, noisy2 = next(iter(train_loader))
    assert tuple(noisy1.shape) == (4, 1, 32, 32)
    assert tuple(noisy2.shape) == (4, 1, 32, 32)

    model = build_model(ModelConfig(arch="noise2noise_unet:n2n_unet_tiny", variant="", in_channels=1, sigma=0.15))
    pred = model(noisy1)
    loss = torch.nn.L1Loss()(pred, noisy2)
    assert torch.isfinite(loss)
    loss.backward()

    noisy, clean = next(iter(val_loader))
    out = model(noisy)
    assert tuple(out.shape) == (4, 1, 32, 32)
    assert tuple(clean.shape) == (4, 1, 32, 32)


def test_vision_denoising_bm3d_forward_smoke() -> None:
    from tracks.vision.lesson_10_synthetic_denoising.data import DataConfig, get_dataloaders
    from tracks.vision.lesson_10_synthetic_denoising.model import ModelConfig, build_model

    _, val_loader = get_dataloaders(
        DataConfig(
            num_samples=32,
            batch_size=2,
            image_size=32,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            in_channels=1,
            noise_std=0.1,
            min_square=6,
            max_square=10,
            train_mode="supervised",
        )
    )
    noisy, clean = next(iter(val_loader))
    model = build_model(ModelConfig(arch="bm3d:bm3d_stage1", variant="", in_channels=1, sigma=0.1))
    out = model(noisy)
    assert tuple(out.shape) == tuple(noisy.shape)
    loss = torch.nn.MSELoss()(out, clean)
    assert torch.isfinite(loss)
