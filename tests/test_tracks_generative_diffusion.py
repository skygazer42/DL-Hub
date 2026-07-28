from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_compact_diffusion_fake_dataloaders_and_noise_prediction_shapes() -> None:
    from tracks.generative.lesson_03_compact_diffusion_mnist.data import DataConfig, get_dataloaders
    from tracks.generative.lesson_03_compact_diffusion_mnist.model import (
        DiffusionMLP,
        DiffusionSchedule,
        ModelConfig,
        q_sample,
    )

    train_loader, val_loader = get_dataloaders(
        DataConfig(
            dataset="fake",
            num_samples=48,
            batch_size=8,
            seed=7,
            num_workers=0,
            val_fraction=0.25,
        )
    )
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    assert train_batch.shape == (8, 1, 28, 28)
    assert val_batch.shape[1:] == (1, 28, 28)

    schedule = DiffusionSchedule(num_steps=12, beta_start=1e-4, beta_end=0.02)
    model = DiffusionMLP(ModelConfig(hidden_dim=64, time_embed_dim=16))
    x0 = train_batch.view(train_batch.size(0), -1)
    timesteps = torch.randint(0, schedule.num_steps, (x0.size(0),), dtype=torch.long)
    noise = torch.randn_like(x0)
    xt = q_sample(schedule, x0, timesteps, noise)

    pred_noise = model(xt, timesteps)
    assert pred_noise.shape == x0.shape
    assert torch.isfinite(pred_noise).all()


def test_compact_diffusion_training_writes_expected_artifacts(tmp_path: Path) -> None:
    from tracks.generative.lesson_03_compact_diffusion_mnist.data import DataConfig
    from tracks.generative.lesson_03_compact_diffusion_mnist.model import ModelConfig
    from tracks.generative.lesson_03_compact_diffusion_mnist.train import TrainConfig, run_training

    run_name = "pytest_diffusion_smoke"
    output_dir = Path("outputs/generative/lesson_03_compact_diffusion_mnist") / run_name
    if output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=3,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            num_sample_steps=4,
            run_name=run_name,
        ),
        DataConfig(
            dataset="fake",
            batch_size=8,
            num_workers=0,
            num_samples=64,
            seed=5,
            val_fraction=0.25,
        ),
        ModelConfig(hidden_dim=64, time_embed_dim=16),
    )

    assert exit_code == 0
    assert (output_dir / "config.json").is_file()
    assert (output_dir / "metrics.jsonl").is_file()
    assert (output_dir / "samples.pt").is_file()
    assert (output_dir / "denoise_grid.pt").is_file()
    assert (output_dir / "checkpoints" / "checkpoint.pt").is_file()
