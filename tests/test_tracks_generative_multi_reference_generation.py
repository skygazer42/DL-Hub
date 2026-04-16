import pytest

torch = pytest.importorskip("torch")


def test_toy_diffusion_multi_reference_generation_data_and_model_contract() -> None:
    from tracks.generative.lesson_29_toy_diffusion_multi_reference_generation.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.generative.lesson_29_toy_diffusion_multi_reference_generation.model import (
        DiffusionSchedule,
        ModelConfig,
        ToyMultiReferenceDiffusionModel,
        q_sample,
    )

    train_loader, _ = get_dataloaders(
        DataConfig(num_samples=48, batch_size=6, image_size=28, seed=0, num_workers=0, val_fraction=0.25)
    )
    reference_a, reference_b, condition, target = next(iter(train_loader))
    assert tuple(reference_a.shape) == (6, 1, 28, 28)
    assert tuple(reference_b.shape) == (6, 1, 28, 28)
    assert tuple(condition.shape) == (6, 1, 28, 28)
    assert tuple(target.shape) == (6, 1, 28, 28)
    assert torch.all(reference_a >= 0.0)
    assert torch.all(reference_a <= 1.0)
    assert torch.all(reference_b >= 0.0)
    assert torch.all(reference_b <= 1.0)
    assert torch.all(condition >= 0.0)
    assert torch.all(condition <= 1.0)
    assert torch.all(target >= 0.0)
    assert torch.all(target <= 1.0)
    assert not torch.allclose(reference_a, target)
    assert not torch.allclose(reference_b, target)
    assert not torch.allclose(condition, target)

    cfg = ModelConfig(image_size=28, in_channels=1, hidden_channels=16, time_embed_dim=16)
    schedule = DiffusionSchedule(num_steps=12)
    model = ToyMultiReferenceDiffusionModel(cfg)

    noise = torch.randn_like(target)
    timesteps = torch.randint(low=0, high=schedule.num_steps, size=(6,), dtype=torch.long)
    xt = q_sample(schedule, target, timesteps, noise)
    pred_noise = model(
        xt=xt,
        reference_a=reference_a,
        reference_b=reference_b,
        condition=condition,
        timesteps=timesteps,
    )
    sampled = model.sample(
        schedule=schedule,
        reference_a=reference_a,
        reference_b=reference_b,
        condition=condition,
        device=torch.device("cpu"),
        num_steps=6,
    )

    assert tuple(pred_noise.shape) == (6, 1, 28, 28)
    assert tuple(sampled.shape) == (6, 1, 28, 28)
    assert torch.all(sampled >= 0.0)
    assert torch.all(sampled <= 1.0)


def test_toy_diffusion_multi_reference_generation_training_smoke(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.generative.lesson_29_toy_diffusion_multi_reference_generation.data import DataConfig
    from tracks.generative.lesson_29_toy_diffusion_multi_reference_generation.model import (
        DiffusionSchedule,
        ModelConfig,
    )
    from tracks.generative.lesson_29_toy_diffusion_multi_reference_generation.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))
    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=11,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_multi_reference_generation_smoke",
            num_sample_steps=6,
        ),
        DataConfig(num_samples=64, batch_size=8, image_size=28, seed=5, num_workers=0, val_fraction=0.25),
        ModelConfig(image_size=28, in_channels=1, hidden_channels=16, time_embed_dim=16),
        DiffusionSchedule(num_steps=12),
    )

    assert exit_code == 0
    run_dir = (
        tmp_path
        / "generative"
        / "lesson_29_toy_diffusion_multi_reference_generation"
        / "pytest_multi_reference_generation_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "samples.pt").is_file()
    assert (run_dir / "multi_reference_trajectory.pt").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()
