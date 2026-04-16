import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_pointcloud_forecasting_batch_contract_and_loss_smoke() -> None:
    from tracks.pointcloud.lesson_32_toy_pointcloud_forecasting.data import (
        DataConfig,
        SyntheticPointCloudForecastingDataset,
        get_dataloaders,
    )
    from tracks.pointcloud.lesson_32_toy_pointcloud_forecasting.model import (
        ModelConfig,
        build_model,
        forecasting_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        sequence_length=4,
        forecast_horizon=2,
        num_points=48,
        batch_size=4,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        motion_scale=0.18,
        jitter_std=0.01,
    )
    ds = SyntheticPointCloudForecastingDataset(cfg)
    history, targets = ds[0]

    assert tuple(history.shape) == (4, 48, 3)
    assert set(targets.keys()) == {"future"}
    assert tuple(targets["future"].shape) == (2, 48, 3)
    assert history.dtype == torch.float32
    assert targets["future"].dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    batch_history, batch_targets = next(iter(train_loader))
    assert tuple(batch_history.shape) == (4, 4, 48, 3)
    assert tuple(batch_targets["future"].shape) == (4, 2, 48, 3)

    model = build_model(
        ModelConfig(
            in_channels=3,
            arch="trajpoint_forecast:trajpoint_forecast_tiny",
            variant="",
            width_mult=1.0,
        )
    )
    outputs = model(batch_history)
    assert set(outputs.keys()) == {"forecast"}
    assert tuple(outputs["forecast"].shape) == (4, 2, 48, 3)

    loss, parts = forecasting_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"forecast_loss", "step_mae"}
    assert float(parts["forecast_loss"]) >= 0.0
    assert float(parts["step_mae"]) >= 0.0
    loss.backward()


def test_pointcloud_forecasting_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.pointcloud.lesson_32_toy_pointcloud_forecasting.data import DataConfig
    from tracks.pointcloud.lesson_32_toy_pointcloud_forecasting.model import ModelConfig
    from tracks.pointcloud.lesson_32_toy_pointcloud_forecasting.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_pointcloud_forecasting_smoke",
            arch="trajpoint_forecast:trajpoint_forecast_tiny",
            width_mult=1.0,
        ),
        DataConfig(
            num_samples=48,
            sequence_length=4,
            forecast_horizon=2,
            num_points=48,
            batch_size=4,
            val_fraction=0.25,
            seed=11,
            num_workers=0,
            motion_scale=0.18,
            jitter_std=0.01,
        ),
        ModelConfig(
            in_channels=3,
            arch="trajpoint_forecast:trajpoint_forecast_tiny",
            variant="",
            width_mult=1.0,
        ),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "pointcloud"
        / "lesson_32_toy_pointcloud_forecasting"
        / "pytest_pointcloud_forecasting_smoke"
    )
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "logs" / "train.log").is_file()
    assert (run_dir / "checkpoints" / "checkpoint.pt").is_file()

    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(metrics) == 1
    record = metrics[0]
    for key in ("train_loss", "train_mae", "eval_loss", "eval_mae"):
        assert key in record
        assert float(record[key]) >= 0.0
