import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_compact_diffusion_reflection_removal_data_and_model_contract() -> None:
    from tracks.generative.lesson_20_compact_diffusion_reflection_removal.data import DataConfig, get_dataloaders
    from tracks.generative.lesson_20_compact_diffusion_reflection_removal.model import (
        DiffusionSchedule,
        ModelConfig,
        CompactReflectionRemovalDiffusionModel,
        q_sample,
    )

    train_loader, _ = get_dataloaders(
        DataConfig(num_samples=48, batch_size=6, image_size=28, seed=0, num_workers=0, val_fraction=0.25)
    )
    reflected, clean = next(iter(train_loader))
    assert tuple(reflected.shape) == (6, 1, 28, 28)
    assert tuple(clean.shape) == (6, 1, 28, 28)
    assert torch.all(reflected >= 0.0)
    assert torch.all(reflected <= 1.0)
    assert torch.all(clean >= 0.0)
    assert torch.all(clean <= 1.0)
    assert not torch.allclose(reflected, clean)

    cfg = ModelConfig(image_size=28, in_channels=1, hidden_channels=16)
    schedule = DiffusionSchedule(num_steps=12)
    model = CompactReflectionRemovalDiffusionModel(cfg)

    noise = torch.randn_like(clean)
    timesteps = torch.randint(low=0, high=schedule.num_steps, size=(6,), dtype=torch.long)
    xt = q_sample(schedule, clean, timesteps, noise)
    pred_noise = model(xt=xt, reflected=reflected, timesteps=timesteps)
    sampled = model.sample(
        schedule=schedule,
        reflected=reflected,
        device=torch.device("cpu"),
        num_steps=6,
    )

    assert tuple(pred_noise.shape) == (6, 1, 28, 28)
    assert tuple(sampled.shape) == (6, 1, 28, 28)
    assert torch.all(sampled >= 0.0)
    assert torch.all(sampled <= 1.0)


def test_compact_diffusion_reflection_removal_training_and_dry_run(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.generative.lesson_20_compact_diffusion_reflection_removal.data import DataConfig
    from tracks.generative.lesson_20_compact_diffusion_reflection_removal.model import DiffusionSchedule, ModelConfig
    from tracks.generative.lesson_20_compact_diffusion_reflection_removal.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))
    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=7,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_reflection_removal_smoke",
            num_sample_steps=6,
        ),
        DataConfig(num_samples=64, batch_size=8, image_size=28, seed=3, num_workers=0, val_fraction=0.25),
        ModelConfig(image_size=28, in_channels=1, hidden_channels=16),
        DiffusionSchedule(num_steps=12),
    )

    assert exit_code == 0
    run_dir = tmp_path / "generative" / "lesson_20_compact_diffusion_reflection_removal" / "pytest_reflection_removal_smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "samples.pt").is_file()
    assert (run_dir / "denoise_grid.pt").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_20_compact_diffusion_reflection_removal",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tracks.generative.lesson_20_compact_diffusion_reflection_removal.train" in proc.stdout
