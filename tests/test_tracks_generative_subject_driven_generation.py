import pytest

torch = pytest.importorskip("torch")


def test_toy_diffusion_subject_driven_generation_data_and_model_contract() -> None:
    from tracks.generative.lesson_28_toy_diffusion_subject_driven_generation.data import (
        DataConfig,
        get_dataloaders,
    )
    from tracks.generative.lesson_28_toy_diffusion_subject_driven_generation.model import (
        DiffusionSchedule,
        ModelConfig,
        ToySubjectDrivenDiffusionModel,
        q_sample,
    )

    train_loader, _ = get_dataloaders(
        DataConfig(num_samples=48, batch_size=6, image_size=28, seed=0, num_workers=0, val_fraction=0.25)
    )
    subject, guidance, target = next(iter(train_loader))
    assert tuple(subject.shape) == (6, 1, 28, 28)
    assert tuple(guidance.shape) == (6, 1, 28, 28)
    assert tuple(target.shape) == (6, 1, 28, 28)
    assert torch.all(subject >= 0.0)
    assert torch.all(subject <= 1.0)
    assert torch.all(guidance >= 0.0)
    assert torch.all(guidance <= 1.0)
    assert torch.all(target >= 0.0)
    assert torch.all(target <= 1.0)
    assert not torch.allclose(subject, target)
    assert not torch.allclose(guidance, target)

    cfg = ModelConfig(image_size=28, in_channels=1, hidden_channels=16, time_embed_dim=16)
    schedule = DiffusionSchedule(num_steps=12)
    model = ToySubjectDrivenDiffusionModel(cfg)

    noise = torch.randn_like(target)
    timesteps = torch.randint(low=0, high=schedule.num_steps, size=(6,), dtype=torch.long)
    xt = q_sample(schedule, target, timesteps, noise)
    pred_noise = model(xt=xt, subject=subject, guidance=guidance, timesteps=timesteps)
    sampled = model.sample(
        schedule=schedule,
        subject=subject,
        guidance=guidance,
        device=torch.device("cpu"),
        num_steps=6,
    )

    assert tuple(pred_noise.shape) == (6, 1, 28, 28)
    assert tuple(sampled.shape) == (6, 1, 28, 28)
    assert torch.all(sampled >= 0.0)
    assert torch.all(sampled <= 1.0)


def test_toy_diffusion_subject_driven_generation_training_smoke(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.generative.lesson_28_toy_diffusion_subject_driven_generation.data import DataConfig
    from tracks.generative.lesson_28_toy_diffusion_subject_driven_generation.model import (
        DiffusionSchedule,
        ModelConfig,
    )
    from tracks.generative.lesson_28_toy_diffusion_subject_driven_generation.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))
    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=7,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_subject_driven_generation_smoke",
            num_sample_steps=6,
        ),
        DataConfig(num_samples=64, batch_size=8, image_size=28, seed=3, num_workers=0, val_fraction=0.25),
        ModelConfig(image_size=28, in_channels=1, hidden_channels=16, time_embed_dim=16),
        DiffusionSchedule(num_steps=12),
    )

    assert exit_code == 0
    run_dir = (
        tmp_path
        / "generative"
        / "lesson_28_toy_diffusion_subject_driven_generation"
        / "pytest_subject_driven_generation_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "samples.pt").is_file()
    assert (run_dir / "conditioning.pt").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()
