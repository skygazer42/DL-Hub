import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_compact_controlnet_dataloaders_smoke() -> None:
    from tracks.generative.lesson_11_compact_controlnet.data import DataConfig, get_dataloaders

    train_loader, val_loader = get_dataloaders(
        DataConfig(num_samples=48, batch_size=8, image_size=28, seed=0, num_workers=0, val_fraction=0.25)
    )
    noisy_target, structure_hint = next(iter(train_loader))
    val_noisy_target, val_structure_hint = next(iter(val_loader))

    assert noisy_target.shape == (8, 1, 28, 28)
    assert structure_hint.shape == (8, 1, 28, 28)
    assert val_noisy_target.shape == (8, 1, 28, 28)
    assert val_structure_hint.shape == (8, 1, 28, 28)
    assert noisy_target.dtype == torch.float32
    assert structure_hint.dtype == torch.float32


def test_compact_controlnet_model_pipeline_smoke() -> None:
    from tracks.generative.lesson_11_compact_controlnet.model import (
        DiffusionSchedule,
        ModelConfig,
        CompactControlNetDenoiser,
        q_sample,
    )

    cfg = ModelConfig(image_size=28, in_channels=1, hidden_channels=24)
    schedule = DiffusionSchedule(num_steps=12)
    model = CompactControlNetDenoiser(cfg)

    target = torch.rand((4, 1, 28, 28), dtype=torch.float32)
    structure_hint = torch.rand((4, 1, 28, 28), dtype=torch.float32)
    noise = torch.randn_like(target)
    timesteps = torch.randint(low=0, high=schedule.num_steps, size=(4,), dtype=torch.long)

    xt = q_sample(schedule, target, timesteps, noise)
    pred_noise = model(xt, structure_hint, timesteps)
    samples = model.sample(schedule, structure_hint=structure_hint, device=torch.device("cpu"), num_steps=6)

    assert xt.shape == target.shape
    assert pred_noise.shape == target.shape
    assert samples.shape == target.shape
    assert torch.all(samples >= 0.0)
    assert torch.all(samples <= 1.0)


def test_compact_controlnet_training_smoke() -> None:
    run_dir = _repo_root() / "outputs" / "generative" / "lesson_11_compact_controlnet" / "pytest_controlnet_smoke"
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.generative.lesson_11_compact_controlnet.train",
            "--epochs",
            "1",
            "--num-samples",
            "48",
            "--batch-size",
            "8",
            "--max-train-batches",
            "2",
            "--max-eval-batches",
            "1",
            "--device",
            "cpu",
            "--run-name",
            "pytest_controlnet_smoke",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "samples.pt").is_file()
    assert (run_dir / "denoise_grid.pt").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()


def test_run_lesson_dry_run_supports_compact_controlnet() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_lesson.py", "generative", "lesson_11_compact_controlnet", "--dry-run"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tracks.generative.lesson_11_compact_controlnet.train" in proc.stdout
