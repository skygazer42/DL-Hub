import pytest

torch = pytest.importorskip("torch")


def test_conditional_gan_batch_contract() -> None:
    from tracks.generative.lesson_09_compact_conditional_gan.data import DataConfig, get_dataloaders
    from tracks.generative.lesson_09_compact_conditional_gan.model import ConditionalGAN, ModelConfig

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=48,
            batch_size=6,
            image_size=28,
            num_classes=4,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
        )
    )
    images, labels = next(iter(train_loader))

    assert tuple(images.shape) == (6, 1, 28, 28)
    assert tuple(labels.shape) == (6,)
    assert torch.all((0 <= labels) & (labels < 4))

    model = ConditionalGAN(
        ModelConfig(z_dim=24, hidden_dim=64, num_classes=4, image_size=28)
    )
    z = torch.randn(6, 24)
    fake = model.generator(z, labels)
    scores = model.discriminator(fake, labels)
    assert tuple(fake.shape) == (6, 1, 28, 28)
    assert tuple(scores.shape) == (6,)


def test_conditional_gan_training_smoke(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.generative.lesson_09_compact_conditional_gan.data import DataConfig
    from tracks.generative.lesson_09_compact_conditional_gan.model import ModelConfig
    from tracks.generative.lesson_09_compact_conditional_gan.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))
    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-4,
            beta1=0.5,
            beta2=0.999,
            seed=7,
            device="cpu",
            max_train_batches=2,
            run_name="pytest_conditional_gan_smoke",
        ),
        DataConfig(
            num_samples=64,
            batch_size=8,
            image_size=28,
            num_classes=4,
            val_fraction=0.25,
            seed=3,
            num_workers=0,
        ),
        ModelConfig(z_dim=24, hidden_dim=64, num_classes=4, image_size=28),
    )

    assert exit_code == 0
    run_dir = tmp_path / "generative" / "lesson_09_compact_conditional_gan" / "pytest_conditional_gan_smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "samples.pt").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

