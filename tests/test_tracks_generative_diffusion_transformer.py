import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_diffusion_transformer_fake_dataloaders_smoke() -> None:
    from tracks.generative.lesson_08_compact_diffusion_transformer.data import DataConfig, get_dataloaders

    train_loader, val_loader = get_dataloaders(
        DataConfig(num_samples=48, batch_size=8, image_size=28, seed=0, num_workers=0, val_fraction=0.25)
    )
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    assert train_batch.shape == (8, 1, 28, 28)
    assert val_batch.shape == (8, 1, 28, 28)
    assert train_batch.dtype == torch.float32
    assert torch.all(train_batch >= 0.0)
    assert torch.all(train_batch <= 1.0)


def test_diffusion_transformer_model_pipeline_smoke() -> None:
    from tracks.generative.lesson_08_compact_diffusion_transformer.model import (
        DiffusionSchedule,
        DiTTiny,
        ModelConfig,
        q_sample,
    )

    cfg = ModelConfig(image_size=28, patch_size=4, in_channels=1, hidden_dim=64, depth=2, num_heads=4)
    schedule = DiffusionSchedule(num_steps=12)
    model = DiTTiny(cfg)

    images = torch.rand((4, 1, 28, 28), dtype=torch.float32)
    noise = torch.randn_like(images)
    timesteps = torch.randint(low=0, high=schedule.num_steps, size=(4,), dtype=torch.long)

    xt = q_sample(schedule, images, timesteps, noise)
    pred_noise = model(xt, timesteps)
    samples = model.sample(schedule, num_samples=4, device=torch.device("cpu"), num_steps=6)

    assert xt.shape == images.shape
    assert pred_noise.shape == images.shape
    assert samples.shape == images.shape
    assert torch.all(samples >= 0.0)
    assert torch.all(samples <= 1.0)


def test_diffusion_transformer_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "generative"
        / "lesson_08_compact_diffusion_transformer"
        / "pytest_diffusion_transformer_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.generative.lesson_08_compact_diffusion_transformer.train",
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
            "pytest_diffusion_transformer_smoke",
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
