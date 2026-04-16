import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_rectified_flow_fake_dataloaders_smoke() -> None:
    from tracks.generative.lesson_07_toy_rectified_flow.data import DataConfig, get_dataloaders

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


def test_rectified_flow_model_pipeline_smoke() -> None:
    from tracks.generative.lesson_07_toy_rectified_flow.model import (
        ModelConfig,
        RectifiedFlowModel,
        build_rectified_targets,
        sample_time,
    )

    cfg = ModelConfig(image_size=28, in_channels=1, hidden_channels=16, time_embed_dim=16)
    model = RectifiedFlowModel(cfg)
    images = torch.rand((4, 1, 28, 28), dtype=torch.float32)
    noise = torch.randn_like(images)
    times = sample_time(batch_size=4, device=images.device)

    xt, target_velocity = build_rectified_targets(images=images, noise=noise, times=times)
    velocity = model(xt, times)
    samples, trajectory = model.sample(
        num_samples=4,
        device=images.device,
        num_steps=8,
        return_trajectory=True,
    )

    assert xt.shape == images.shape
    assert target_velocity.shape == images.shape
    assert velocity.shape == images.shape
    assert samples.shape == images.shape
    assert trajectory.shape == (9, 4, 1, 28, 28)
    assert torch.isfinite(velocity).all()


def test_rectified_flow_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "generative"
        / "lesson_07_toy_rectified_flow"
        / "pytest_rectified_flow_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.generative.lesson_07_toy_rectified_flow.train",
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
            "pytest_rectified_flow_smoke",
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
    assert (run_dir / "interp.pt").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()
