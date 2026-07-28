import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")


def test_compact_diffusion_style_transfer_data_and_model_contract() -> None:
    from tracks.generative.lesson_22_compact_diffusion_style_transfer.data import DataConfig, get_dataloaders
    from tracks.generative.lesson_22_compact_diffusion_style_transfer.model import (
        DiffusionSchedule,
        ModelConfig,
        CompactStyleTransferDiffusionModel,
        q_sample,
    )

    train_loader, _ = get_dataloaders(
        DataConfig(num_samples=48, batch_size=6, image_size=28, seed=0, num_workers=0, val_fraction=0.25)
    )
    content, style, target = next(iter(train_loader))
    assert tuple(content.shape) == (6, 1, 28, 28)
    assert tuple(style.shape) == (6, 1, 28, 28)
    assert tuple(target.shape) == (6, 1, 28, 28)
    assert torch.all(content >= 0.0)
    assert torch.all(style >= 0.0)
    assert torch.all(target >= 0.0)
    assert torch.all(target <= 1.0)
    assert not torch.allclose(content, target)

    cfg = ModelConfig(image_size=28, in_channels=1, hidden_channels=16)
    schedule = DiffusionSchedule(num_steps=12)
    model = CompactStyleTransferDiffusionModel(cfg)

    noise = torch.randn_like(target)
    timesteps = torch.randint(low=0, high=schedule.num_steps, size=(6,), dtype=torch.long)
    xt = q_sample(schedule, target, timesteps, noise)
    pred_noise = model(xt=xt, content=content, style=style, timesteps=timesteps)
    sampled = model.sample(
        schedule=schedule,
        content=content,
        style=style,
        device=torch.device("cpu"),
        num_steps=6,
    )

    assert tuple(pred_noise.shape) == (6, 1, 28, 28)
    assert tuple(sampled.shape) == (6, 1, 28, 28)
    assert torch.all(sampled >= 0.0)
    assert torch.all(sampled <= 1.0)


def test_compact_diffusion_style_transfer_training_and_dry_run(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.generative.lesson_22_compact_diffusion_style_transfer.data import DataConfig
    from tracks.generative.lesson_22_compact_diffusion_style_transfer.model import DiffusionSchedule, ModelConfig
    from tracks.generative.lesson_22_compact_diffusion_style_transfer.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))
    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=7,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_style_transfer_smoke",
            num_sample_steps=6,
        ),
        DataConfig(num_samples=64, batch_size=8, image_size=28, seed=3, num_workers=0, val_fraction=0.25),
        ModelConfig(image_size=28, in_channels=1, hidden_channels=16),
        DiffusionSchedule(num_steps=12),
    )

    assert exit_code == 0
    run_dir = tmp_path / "generative" / "lesson_22_compact_diffusion_style_transfer" / "pytest_style_transfer_smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "samples.pt").is_file()
    assert (run_dir / "style_trajectory.pt").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_22_compact_diffusion_style_transfer",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tracks.generative.lesson_22_compact_diffusion_style_transfer.train" in proc.stdout
