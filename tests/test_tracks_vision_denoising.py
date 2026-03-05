import pytest


torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    "noise_type,kwargs",
    [
        ("gaussian", {}),
        ("gaussian_var", {"noise_std_min": 0.05, "noise_std_max": 0.2}),
        ("gaussian_impulse", {"impulse_prob": 0.03}),
        ("poisson", {"poisson_peak": 30.0}),
        ("poisson_gaussian", {"poisson_peak": 30.0, "read_noise": 0.02}),
        ("impulse", {"impulse_prob": 0.05}),
        ("clustered_impulse", {"cluster_prob": 0.002, "cluster_size": 5, "impulse_prob": 0.8}),
        ("shot_read", {"shot_noise": 0.2, "read_noise": 0.02}),
        ("speckle", {"speckle_std": 0.15}),
        ("speckle_read", {"speckle_std": 0.15, "read_noise": 0.02}),
        ("stripe", {"stripe_amplitude": 0.12, "stripe_period": 8}),
        ("stripe", {"stripe_amplitude": 0.12, "stripe_period": 8, "stripe_direction": "random"}),
        ("block_bias", {"block_size": 8, "block_std": 0.05}),
        ("correlated_gaussian", {}),
        ("quantization", {"quant_bits": 6, "quant_dither": False}),
        ("dead_hot", {"defect_prob": 0.01, "defect_hot_ratio": 0.3}),
        ("rowcol_bias", {"row_bias_std": 0.02, "col_bias_std": 0.02}),
        ("mixed", {"shot_noise": 0.2, "read_noise": 0.02, "impulse_prob": 0.03, "quant_bits": 8}),
        ("colored_gaussian", {"color_rho": 0.6}),
        ("line_defect", {"line_prob": 0.03, "line_hot_ratio": 0.3}),
    ],
)
def test_vision_denoising_noise_models_dataloader_smoke(noise_type: str, kwargs: dict) -> None:
    from tracks.vision.lesson_10_synthetic_denoising.data import DataConfig, get_dataloaders

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=32,
            batch_size=4,
            image_size=32,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            in_channels=1,
            noise_type=noise_type,
            noise_std=0.15,
            train_mode="supervised",
            min_square=6,
            max_square=10,
            **kwargs,
        )
    )
    noisy, clean = next(iter(train_loader))
    assert tuple(noisy.shape) == (4, 1, 32, 32)
    assert tuple(clean.shape) == (4, 1, 32, 32)
    assert noisy.min().item() >= 0.0
    assert noisy.max().item() <= 1.0


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
        "edsr:edsr_tiny",
        "rrdbnet:rrdbnet_tiny",
        "carn:carn_tiny",
        "resunet:resunet_tiny",
        "unet3plus:unet3plus_tiny",
        "r2unet:r2unet_tiny",
        "denseunet:denseunet_tiny",
        "brdnet:brdnet_tiny",
        "attention_unet:attention_unet_tiny",
        "unetpp:unetpp_tiny",
        "mwcnn:mwcnn_tiny",
        "hinet:hinet_tiny",
        "ircnn:ircnn_tiny",
        "nlrn:nlrn_tiny",
        "scunet:scunet_tiny",
        "convnext_unet:convnext_unet_tiny",
        "aspp_unet:aspp_unet_tiny",
        "cbam_unet:cbam_unet_tiny",
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
        "cbdnet:cbdnet_tiny",
        "didn:didn_tiny",
        "rcan:rcan_tiny",
        "rdn:rdn_tiny",
        "memnet:memnet_tiny",
        "drrn:drrn_tiny",
        "rednet:rednet_tiny",
        "pridnet:pridnet_tiny",
        "dhdn:dhdn_tiny",
        "bsn:bsn_tiny",
        "pixelcnn_bsn:pixelcnn_bsn_tiny",
        "dbsn:dbsn_tiny",
        "gated_pixelcnn_bsn:gated_pixelcnn_bsn_tiny",
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


def test_vision_denoising_blindspot_fit_regression_smoke() -> None:
    from dlhub.training.loop import fit_regression
    from tracks.vision.lesson_10_synthetic_denoising.data import DataConfig, get_dataloaders
    from tracks.vision.lesson_10_synthetic_denoising.losses import MaskedMSELoss
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
            noise_type="poisson",
            poisson_peak=30.0,
            min_square=6,
            max_square=10,
            train_mode="blindspot",
            blindspot_prob=0.1,
        )
    )
    masked_noisy, target = next(iter(train_loader))
    assert tuple(masked_noisy.shape) == (4, 1, 32, 32)
    assert set(target.keys()) == {"target", "mask"}
    assert tuple(target["target"].shape) == (4, 1, 32, 32)
    assert tuple(target["mask"].shape) == (4, 1, 32, 32)

    model = build_model(ModelConfig(arch="dncnn:dncnn_tiny", variant="", in_channels=1, sigma=0.15))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    stats = fit_regression(
        model=model,
        loader=train_loader,
        optimizer=opt,
        criterion=MaskedMSELoss(),
        device=torch.device("cpu"),
        max_batches=1,
    )
    assert stats.loss >= 0.0


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


@pytest.mark.parametrize(
    "arch",
    [
        "median_filter:median_tiny",
        "wiener_filter:wiener_tiny",
        "guided_filter:guided_filter_tiny",
        "bilateral_filter:bilateral_fast",
        "non_local_means:nlm_fast",
        "total_variation:tv_fast",
        "anisotropic_diffusion:anisodiff_fast",
        "wavelet_shrinkage:wavelet_tiny",
    ],
)
def test_vision_denoising_classical_baselines_forward_smoke(arch: str) -> None:
    from tracks.vision.lesson_10_synthetic_denoising.model import ModelConfig, build_model

    torch.manual_seed(0)
    clean = torch.rand(2, 1, 32, 32)
    noisy = (clean + torch.randn_like(clean) * 0.12).clamp(0.0, 1.0)

    model = build_model(ModelConfig(arch=arch, variant="", in_channels=1, sigma=0.12))
    out = model(noisy)
    assert tuple(out.shape) == tuple(noisy.shape)

    loss = torch.nn.MSELoss()(out, clean)
    assert torch.isfinite(loss)
