import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_consistency_model_fake_dataloaders_smoke() -> None:
    from tracks.generative.lesson_05_toy_consistency_model.data import DataConfig, get_dataloaders

    train_loader, val_loader = get_dataloaders(
        DataConfig(num_samples=48, batch_size=8, seed=0, num_workers=0, val_fraction=0.25)
    )
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    assert train_batch.shape == (8, 1, 28, 28)
    assert val_batch.shape == (8, 1, 28, 28)
    assert train_batch.dtype == torch.float32
    assert torch.all(train_batch >= 0.0)
    assert torch.all(train_batch <= 1.0)


def test_consistency_model_boundary_condition_and_loss() -> None:
    from tracks.generative.lesson_05_toy_consistency_model.model import (
        ConsistencyModel,
        ConsistencySchedule,
        ModelConfig,
        consistency_training_loss,
        update_ema,
    )

    schedule = ConsistencySchedule(num_steps=6, sigma_min=0.02, sigma_max=3.0)
    sigmas = schedule.sigmas()
    assert sigmas.shape == (6,)
    assert torch.all(sigmas[1:] > sigmas[:-1])
    assert sigmas[0] == pytest.approx(schedule.sigma_min, rel=1e-5)
    assert sigmas[-1] == pytest.approx(schedule.sigma_max, rel=1e-5)

    model = ConsistencyModel(ModelConfig(hidden_dim=16, time_embed_dim=8), schedule)
    x = torch.rand((4, 28 * 28), dtype=torch.float32)

    # f(x, sigma_min) must be the identity by construction.
    at_boundary = model(x, torch.full((4,), schedule.sigma_min))
    assert torch.allclose(at_boundary, x, atol=1e-5)

    target_model = ConsistencyModel(ModelConfig(hidden_dim=16, time_embed_dim=8), schedule)
    target_model.load_state_dict(model.state_dict())
    loss = consistency_training_loss(model, target_model, x, schedule)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())

    update_ema(target_model, model, decay=0.5)


def test_consistency_model_sampling_smoke() -> None:
    from tracks.generative.lesson_05_toy_consistency_model.model import (
        ConsistencyModel,
        ConsistencySchedule,
        ModelConfig,
        sample_consistency,
    )

    schedule = ConsistencySchedule(num_steps=6)
    model = ConsistencyModel(ModelConfig(hidden_dim=16, time_embed_dim=8), schedule)

    one_step = sample_consistency(model, schedule, num_samples=4, device=torch.device("cpu"))
    frames = sample_consistency(
        model,
        schedule,
        num_samples=4,
        device=torch.device("cpu"),
        num_steps=3,
        return_all=True,
    )

    assert one_step.shape == (4, 1, 28, 28)
    assert torch.all(one_step >= 0.0)
    assert torch.all(one_step <= 1.0)
    assert frames.shape == (3, 4, 1, 28, 28)


def test_consistency_model_training_smoke() -> None:
    run_dir = (
        _repo_root()
        / "outputs"
        / "generative"
        / "lesson_05_toy_consistency_model"
        / "pytest_consistency_model_smoke"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracks.generative.lesson_05_toy_consistency_model.train",
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
            "--num-discretization-steps",
            "6",
            "--device",
            "cpu",
            "--run-name",
            "pytest_consistency_model_smoke",
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
    assert (run_dir / "refine_grid.pt").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()
