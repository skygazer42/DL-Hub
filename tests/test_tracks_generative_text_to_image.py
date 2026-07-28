import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_compact_text_to_image_dataloaders_smoke() -> None:
    from tracks.generative.lesson_13_compact_text_to_image_diffusion.data import DataConfig, get_dataloaders

    train_loader, val_loader = get_dataloaders(
        DataConfig(num_samples=48, batch_size=8, image_size=28, seed=0, num_workers=0, val_fraction=0.25)
    )
    token_ids, images = next(iter(train_loader))
    val_token_ids, val_images = next(iter(val_loader))

    assert token_ids.shape == (8,)
    assert images.shape == (8, 1, 28, 28)
    assert val_token_ids.shape == (8,)
    assert val_images.shape == (8, 1, 28, 28)
    assert token_ids.dtype == torch.long
    assert images.dtype == torch.float32
    assert torch.all(images >= 0.0)
    assert torch.all(images <= 1.0)


def test_compact_text_to_image_model_pipeline_smoke() -> None:
    from tracks.generative.lesson_13_compact_text_to_image_diffusion.model import (
        DiffusionSchedule,
        ModelConfig,
        CompactTextConditionedDenoiser,
        q_sample,
    )

    cfg = ModelConfig(image_size=28, in_channels=1, hidden_channels=24, text_vocab_size=4)
    schedule = DiffusionSchedule(num_steps=12)
    model = CompactTextConditionedDenoiser(cfg)

    images = torch.rand((4, 1, 28, 28), dtype=torch.float32)
    token_ids = torch.randint(low=0, high=4, size=(4,), dtype=torch.long)
    noise = torch.randn_like(images)
    timesteps = torch.randint(low=0, high=schedule.num_steps, size=(4,), dtype=torch.long)

    xt = q_sample(schedule, images, timesteps, noise)
    pred_noise = model(xt, token_ids, timesteps)
    samples = model.sample(schedule, token_ids=token_ids, device=torch.device("cpu"), num_steps=6)

    assert xt.shape == images.shape
    assert pred_noise.shape == images.shape
    assert samples.shape == images.shape
    assert torch.all(samples >= 0.0)
    assert torch.all(samples <= 1.0)


def test_compact_text_to_image_training_smoke() -> None:
    run_dir = (
        _repo_root() / "outputs" / "generative" / "lesson_13_compact_text_to_image_diffusion" / "pytest_text_to_image_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.generative.lesson_13_compact_text_to_image_diffusion.train",
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
            "pytest_text_to_image_smoke",
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


def test_run_lesson_dry_run_supports_compact_text_to_image() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_lesson.py", "generative", "lesson_13_compact_text_to_image_diffusion", "--dry-run"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "tracks.generative.lesson_13_compact_text_to_image_diffusion.train" in proc.stdout
