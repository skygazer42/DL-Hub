from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_synthetic_crowd_counting_shapes_and_loss_smoke() -> None:
    from tracks.vision.lesson_18_synthetic_crowd_counting.data import DataConfig, get_dataloaders
    from tracks.vision.lesson_18_synthetic_crowd_counting.model import (
        CrowdCountingRegressor,
        ModelConfig,
    )
    from tracks.vision.lesson_18_synthetic_crowd_counting.train import compute_count_metrics

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=32,
            batch_size=4,
            image_size=64,
            val_fraction=0.25,
            seed=0,
            num_workers=0,
            min_people=2,
            max_people=6,
        )
    )
    x, density, count = next(iter(train_loader))

    assert tuple(x.shape) == (4, 1, 64, 64)
    assert tuple(density.shape) == (4, 1, 64, 64)
    assert tuple(count.shape) == (4,)
    assert density.dtype == torch.float32
    assert count.dtype == torch.float32
    assert torch.allclose(density.sum(dim=(1, 2, 3)), count, atol=1e-3)

    model = CrowdCountingRegressor(ModelConfig(in_channels=1, hidden_channels=16, depth=3))
    pred_density = model(x)
    assert tuple(pred_density.shape) == (4, 1, 64, 64)

    loss = torch.nn.functional.mse_loss(pred_density, density)
    assert torch.isfinite(loss)

    metrics = compute_count_metrics(pred_density.detach(), density)
    assert set(metrics) == {"count_mae", "count_rmse", "pred_count_mean", "target_count_mean"}
    assert torch.isfinite(torch.tensor(metrics["count_mae"]))
    assert torch.isfinite(torch.tensor(metrics["count_rmse"]))

    loss.backward()


def test_vision_synthetic_crowd_counting_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_18_synthetic_crowd_counting.train import (
        DataConfig,
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    tc = TrainConfig(
        epochs=1,
        learning_rate=1e-3,
        seed=0,
        device="cpu",
        max_train_batches=1,
        max_eval_batches=1,
        run_name="smoke",
        hidden_channels=16,
        depth=3,
        dropout=0.0,
    )
    dc = DataConfig(
        num_samples=32,
        batch_size=4,
        image_size=64,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        min_people=2,
        max_people=6,
    )

    run_training(tc, dc)

    run_dir = tmp_path / "vision" / "lesson_18_synthetic_crowd_counting" / "smoke"
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(metrics) == 1
    record = metrics[0]
    for key in ("train_loss", "train_count_mae", "eval_loss", "eval_count_mae", "eval_count_rmse"):
        assert key in record
        assert float(record[key]) >= 0.0
