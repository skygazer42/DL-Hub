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
        (
            "rain",
            {
                "rain_count": 24,
                "rain_length_min": 8,
                "rain_length_max": 18,
                "rain_intensity_min": 0.05,
                "rain_intensity_max": 0.14,
            },
        ),
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
        "jorder:jorder_tiny",
        "rescan:rescan_tiny",
        "prenet:prenet_tiny",
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


@pytest.mark.parametrize(
    "arch",
    [
        "ddn:ddn_tiny",
        "spanet:spanet_tiny",
        "did_mdn:did_mdn_tiny",
        "rcdnet:rcdnet_tiny",
        "transweather:transweather_tiny",
        "derainformer:derainformer_tiny",
    ],
)
def test_vision_denoising_deraining_rain_models_forward_backward_smoke(arch: str) -> None:
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
            noise_type="rain",
            rain_count=24,
            rain_length_min=8,
            rain_length_max=18,
            rain_intensity_min=0.05,
            rain_intensity_max=0.14,
            train_mode="supervised",
        )
    )
    noisy, clean = next(iter(train_loader))
    model = build_model(ModelConfig(arch=arch, variant="", in_channels=1, sigma=0.15))
    pred = model(noisy)
    assert tuple(pred.shape) == tuple(clean.shape)
    loss = torch.nn.MSELoss()(pred, clean)
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize(
    "arch",
    [
        "transweather:transweather_tiny",
        "derainformer:derainformer_tiny",
    ],
)
@pytest.mark.parametrize("size", [1, 2])
def test_vision_denoising_transformer_derainers_reject_tiny_inputs(
    arch: str, size: int
) -> None:
    from tracks.vision.lesson_10_synthetic_denoising.model import ModelConfig, build_model

    model = build_model(ModelConfig(arch=arch, variant="", in_channels=1, sigma=0.15))
    noisy = torch.rand(2, 1, size, size)

    with pytest.raises(ValueError, match="too small for reflect padding"):
        model(noisy)


@pytest.mark.parametrize(
    "arch",
    [
        "transweather:transweather_tiny",
        "derainformer:derainformer_tiny",
    ],
)
def test_vision_denoising_transformer_derainers_accept_smallest_valid_inputs(
    arch: str,
) -> None:
    from tracks.vision.lesson_10_synthetic_denoising.model import ModelConfig, build_model

    model = build_model(ModelConfig(arch=arch, variant="", in_channels=1, sigma=0.15))
    noisy = torch.rand(2, 1, 3, 3)

    out = model(noisy)

    assert tuple(out.shape) == tuple(noisy.shape)
    assert torch.isfinite(out).all()


def test_vision_denoising_deraining_package_exports_smoke() -> None:
    from dlhub.vision.denoising import (
        DDN,
        DIDMDN,
        RCDNet,
        SPANet,
        DerainFormer,
        TransWeather,
        build_ddn_denoiser,
        build_derainformer_denoiser,
        build_did_mdn_denoiser,
        build_rcdnet_denoiser,
        build_spanet_denoiser,
        build_transweather_denoiser,
    )

    specs = [
        (DDN, build_ddn_denoiser, "ddn_tiny"),
        (SPANet, build_spanet_denoiser, "spanet_tiny"),
        (DIDMDN, build_did_mdn_denoiser, "did_mdn_tiny"),
        (RCDNet, build_rcdnet_denoiser, "rcdnet_tiny"),
        (TransWeather, build_transweather_denoiser, "transweather_tiny"),
        (DerainFormer, build_derainformer_denoiser, "derainformer_tiny"),
    ]

    for cls, builder, variant in specs:
        model = builder(in_channels=1, variant=variant)
        assert isinstance(model, cls)


@pytest.mark.parametrize(
    ("arch", "variant"),
    [
        ("spa_net", "spanet_tiny"),
        ("did-mdn", "did_mdn_tiny"),
        ("didmdn", "did_mdn_tiny"),
        ("trans_weather", "transweather_tiny"),
        ("derain_former", "derainformer_tiny"),
    ],
)
def test_vision_denoising_deraining_alias_arches_forward_smoke(
    arch: str, variant: str
) -> None:
    from tracks.vision.lesson_10_synthetic_denoising.model import ModelConfig, build_model

    model = build_model(ModelConfig(arch=f"{arch}:{variant}", variant="", in_channels=1, sigma=0.15))
    noisy = torch.rand(2, 1, 8, 8)

    out = model(noisy)

    assert tuple(out.shape) == tuple(noisy.shape)


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

    model = build_model(
        ModelConfig(arch="noise2noise_unet:n2n_unet_tiny", variant="", in_channels=1, sigma=0.15)
    )
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
        "anscombe_wiener:anscombe_wiener_tiny",
        "lee_filter:lee_tiny",
        "kuan_filter:kuan_tiny",
        "stripe_remover:stripe_remover_tiny",
        "dead_hot_pixel_corrector:dead_hot_tiny",
        "line_defect_corrector:line_defect_tiny",
        "rowcol_bias_corrector:rowcol_bias_tiny",
        "block_bias_corrector:block_bias_tiny",
        "debanding_filter:deband_tiny",
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


def test_vision_denoising_defect_baselines_reduce_error_smoke() -> None:
    from tracks.vision.lesson_10_synthetic_denoising.model import ModelConfig, build_model

    torch.manual_seed(0)

    # --- dead/hot pixel correction
    clean = torch.zeros(1, 1, 32, 32)
    clean[:, :, 8:24, 8:24] = 1.0
    noisy = clean.clone()
    noisy[:, :, 5, 5] = 1.0
    noisy[:, :, 6, 7] = 1.0
    noisy[:, :, 10, 10] = 0.0

    model = build_model(
        ModelConfig(
            arch="dead_hot_pixel_corrector:dead_hot_tiny", variant="", in_channels=1, sigma=0.1
        )
    )
    out = model(noisy)
    assert (out - clean).abs().mean().item() < (noisy - clean).abs().mean().item()

    # --- stuck rows/cols correction
    clean2 = clean.clone()
    noisy2 = clean2.clone()
    noisy2[:, :, 3, :] = 1.0  # hot row
    noisy2[:, :, :, 4] = 1.0  # hot col

    model = build_model(
        ModelConfig(
            arch="line_defect_corrector:line_defect_tiny", variant="", in_channels=1, sigma=0.1
        )
    )
    out2 = model(noisy2)
    assert (out2 - clean2).abs().mean().item() < (noisy2 - clean2).abs().mean().item()

    # --- row/col fixed-pattern bias correction
    clean3 = torch.full((1, 1, 32, 32), 0.5)
    row_bias = torch.randn(1, 1, 32, 1) * 0.04
    col_bias = torch.randn(1, 1, 1, 32) * 0.04
    noisy3 = (clean3 + row_bias + col_bias).clamp(0.0, 1.0)

    model = build_model(
        ModelConfig(
            arch="rowcol_bias_corrector:rowcol_bias_tiny", variant="", in_channels=1, sigma=0.1
        )
    )
    out3 = model(noisy3)
    assert (out3 - clean3).abs().mean().item() < (noisy3 - clean3).abs().mean().item()

    # --- block-wise bias correction
    clean4 = torch.full((1, 1, 32, 32), 0.5)
    block_bias = torch.randn(1, 1, 4, 4) * 0.05
    block_bias = block_bias.repeat_interleave(8, dim=-2).repeat_interleave(8, dim=-1)
    noisy4 = (clean4 + block_bias).clamp(0.0, 1.0)

    model = build_model(
        ModelConfig(
            arch="block_bias_corrector:block_bias_tiny", variant="", in_channels=1, sigma=0.1
        )
    )
    out4 = model(noisy4)
    assert (out4 - clean4).abs().mean().item() < (noisy4 - clean4).abs().mean().item()

    # --- debanding / dequantization
    h, w = 64, 64
    ramp = torch.linspace(0.0, 1.0, w).view(1, 1, 1, w).expand(1, 1, h, w)
    bits = 5
    levels = float((1 << bits) - 1)
    noisy5 = torch.round(ramp * levels) / levels

    model = build_model(
        ModelConfig(arch="debanding_filter:deband_tiny", variant="", in_channels=1, sigma=0.1)
    )
    out5 = model(noisy5)
    assert (out5 - ramp).abs().mean().item() < (noisy5 - ramp).abs().mean().item()


def test_vision_denoising_train_parse_args_wires_clustered_impulse_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from tracks.vision.lesson_10_synthetic_denoising.train import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--noise-type",
            "clustered_impulse",
            "--cluster-prob",
            "0.123",
            "--cluster-size",
            "7",
        ],
    )
    _, data_cfg = parse_args()
    assert data_cfg.noise_type == "clustered_impulse"
    assert data_cfg.cluster_prob == pytest.approx(0.123)
    assert data_cfg.cluster_size == 7


def test_vision_denoising_train_parse_args_wires_block_bias_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from tracks.vision.lesson_10_synthetic_denoising.train import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--noise-type",
            "block_bias",
            "--block-size",
            "4",
            "--block-std",
            "0.07",
        ],
    )
    _, data_cfg = parse_args()
    assert data_cfg.noise_type == "block_bias"
    assert data_cfg.block_size == 4
    assert data_cfg.block_std == pytest.approx(0.07)


def test_vision_denoising_train_parse_args_wires_rain_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from tracks.vision.lesson_10_synthetic_denoising.train import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--noise-type",
            "rain",
            "--rain-count",
            "12",
            "--rain-length-min",
            "6",
            "--rain-length-max",
            "14",
            "--rain-intensity-min",
            "0.04",
            "--rain-intensity-max",
            "0.12",
            "--rain-angle-deg",
            "80",
            "--rain-angle-jitter-deg",
            "8",
        ],
    )
    _, data_cfg = parse_args()
    assert data_cfg.noise_type == "rain"
    assert data_cfg.rain_count == 12
    assert data_cfg.rain_length_min == 6
    assert data_cfg.rain_length_max == 14
    assert data_cfg.rain_intensity_min == pytest.approx(0.04)
    assert data_cfg.rain_intensity_max == pytest.approx(0.12)
    assert data_cfg.rain_angle_deg == pytest.approx(80.0)
    assert data_cfg.rain_angle_jitter_deg == pytest.approx(8.0)


def test_vision_denoising_train_parse_args_list_noise_types(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import sys

    from tracks.vision.lesson_10_synthetic_denoising.train import parse_args

    monkeypatch.setattr(sys, "argv", ["prog", "--list-noise-types"])

    with pytest.raises(SystemExit) as exc:
        parse_args()

    assert exc.value.code == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert "gaussian" in out
    assert "mixed" in out
    assert len(out) >= 19


def test_vision_denoising_train_parse_args_list_arch_families(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import sys

    from tracks.vision.lesson_10_synthetic_denoising.train import parse_args

    monkeypatch.setattr(sys, "argv", ["prog", "--list-arch-families"])

    with pytest.raises(SystemExit) as exc:
        parse_args()

    assert exc.value.code == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert "dncnn" in out
    assert "bm3d" in out
    assert "jorder" in out
    assert "rescan" in out
    assert "prenet" in out
    assert "ddn" in out
    assert "spanet" in out
    assert "did_mdn" in out
    assert "rcdnet" in out
    assert "transweather" in out
    assert "derainformer" in out
    assert len(out) >= 10


def test_vision_denoising_train_parse_args_list_arch_filters_by_family(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import sys

    from tracks.vision.lesson_10_synthetic_denoising.train import parse_args

    monkeypatch.setattr(sys, "argv", ["prog", "--list-arch", "--arch-family", "dncnn"])

    with pytest.raises(SystemExit) as exc:
        parse_args()

    assert exc.value.code == 0
    out = [ln.strip() for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert any(ln.startswith("dncnn:") for ln in out)
    assert all(ln.startswith("dncnn:") for ln in out)

    monkeypatch.setattr(sys, "argv", ["prog", "--list-arch", "--arch-family", "ddn"])

    with pytest.raises(SystemExit) as exc2:
        parse_args()

    assert exc2.value.code == 0
    out2 = [ln.strip() for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert any(ln.startswith("ddn:") for ln in out2)
    assert all(ln.startswith("ddn:") for ln in out2)


def test_vision_denoising_train_parse_args_list_arch_filters_by_match(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import sys

    from tracks.vision.lesson_10_synthetic_denoising.train import parse_args

    monkeypatch.setattr(sys, "argv", ["prog", "--list-arch", "--arch-match", "tiny"])

    with pytest.raises(SystemExit) as exc:
        parse_args()

    assert exc.value.code == 0
    out = [ln.strip() for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert any(ln == "dncnn:dncnn_tiny" for ln in out)
    assert all("tiny" in ln for ln in out)


def test_vision_denoising_train_parse_args_list_arch_respects_list_limit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import sys

    from tracks.vision.lesson_10_synthetic_denoising.train import parse_args

    monkeypatch.setattr(
        sys, "argv", ["prog", "--list-arch", "--arch-family", "dncnn", "--list-limit", "2"]
    )

    with pytest.raises(SystemExit) as exc:
        parse_args()

    assert exc.value.code == 0
    out = [ln.strip() for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(out) == 2
    assert all(ln.startswith("dncnn:") for ln in out)


def test_vision_denoising_train_parse_args_arch_family_requires_list_arch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import sys

    from tracks.vision.lesson_10_synthetic_denoising.train import parse_args

    monkeypatch.setattr(sys, "argv", ["prog", "--arch-family", "dncnn"])

    with pytest.raises(SystemExit) as exc:
        parse_args()

    assert exc.value.code == 2
    assert "--arch-family" in capsys.readouterr().err


def test_vision_denoising_train_parse_args_arch_match_requires_list_arch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import sys

    from tracks.vision.lesson_10_synthetic_denoising.train import parse_args

    monkeypatch.setattr(sys, "argv", ["prog", "--arch-match", "tiny"])

    with pytest.raises(SystemExit) as exc:
        parse_args()

    assert exc.value.code == 2
    assert "--arch-match" in capsys.readouterr().err


def test_vision_denoising_train_parse_args_list_limit_requires_list_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import sys

    from tracks.vision.lesson_10_synthetic_denoising.train import parse_args

    monkeypatch.setattr(sys, "argv", ["prog", "--list-limit", "3"])

    with pytest.raises(SystemExit) as exc:
        parse_args()

    assert exc.value.code == 2
    assert "--list-limit" in capsys.readouterr().err


def test_vision_denoising_train_parse_args_list_noise_types_sorts_alpha(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import sys

    from tracks.vision.lesson_10_synthetic_denoising.train import parse_args

    monkeypatch.setattr(sys, "argv", ["prog", "--list-noise-types", "--list-sort", "alpha"])

    with pytest.raises(SystemExit) as exc:
        parse_args()

    assert exc.value.code == 0
    out = [ln.strip() for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert out == sorted(out)


def test_vision_denoising_train_parse_args_list_sort_requires_list_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import sys

    from tracks.vision.lesson_10_synthetic_denoising.train import parse_args

    monkeypatch.setattr(sys, "argv", ["prog", "--list-sort", "alpha"])

    with pytest.raises(SystemExit) as exc:
        parse_args()

    assert exc.value.code == 2
    assert "--list-sort" in capsys.readouterr().err


def test_vision_denoising_train_parse_args_print_config_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import json
    import sys

    from tracks.vision.lesson_10_synthetic_denoising.train import parse_args

    monkeypatch.setattr(
        sys, "argv", ["prog", "--print-config", "--epochs", "3", "--noise-type", "gaussian"]
    )

    with pytest.raises(SystemExit) as exc:
        parse_args()

    assert exc.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["train"]["epochs"] == 3
    assert payload["data"]["noise_type"] == "gaussian"


def test_vision_denoising_cli_list_noise_types_does_not_print_pynvml_warning() -> None:
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.vision.lesson_10_synthetic_denoising.train",
            "--list-noise-types",
        ],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "pynvml package is deprecated" not in proc.stderr


def test_vision_denoising_cli_print_config_does_not_print_pynvml_warning_and_is_json() -> None:
    import json
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.vision.lesson_10_synthetic_denoising.train",
            "--print-config",
            "--epochs",
            "1",
            "--noise-type",
            "gaussian",
        ],
        cwd=str(repo_root),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "pynvml package is deprecated" not in proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["train"]["epochs"] == 1
