import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_image_deweathering_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_76_synthetic_image_deweathering.data import (
        DataConfig,
        SyntheticImageDeweatheringDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_76_synthetic_image_deweathering.model import (
        ModelConfig,
        build_model,
        deweathering_loss,
        list_supported_arches,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        image_size=32,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=3,
        streak_count_min=5,
        streak_count_max=10,
        weather_strength_min=0.12,
        weather_strength_max=0.28,
    )
    ds = SyntheticImageDeweatheringDataset(cfg)
    weathered, targets = ds[0]

    assert tuple(weathered.shape) == (3, 32, 32)
    assert set(targets.keys()) == {"clean", "weather_residual"}
    assert tuple(targets["clean"].shape) == (3, 32, 32)
    assert tuple(targets["weather_residual"].shape) == (3, 32, 32)
    assert weathered.dtype == torch.float32
    assert targets["clean"].dtype == torch.float32
    assert targets["weather_residual"].dtype == torch.float32
    assert "deweather:deweather_cnn_tiny" in list_supported_arches()

    train_loader, _ = get_dataloaders(cfg)
    batch_weathered, batch_targets = next(iter(train_loader))
    assert tuple(batch_weathered.shape) == (4, 3, 32, 32)
    assert tuple(batch_targets["clean"].shape) == (4, 3, 32, 32)
    assert tuple(batch_targets["weather_residual"].shape) == (4, 3, 32, 32)

    model = build_model(
        ModelConfig(
            in_channels=3,
            arch="deweather:deweather_cnn_tiny",
            width_mult=1.0,
        )
    )
    outputs = model(batch_weathered)
    assert set(outputs.keys()) == {"restored", "weather_residual"}
    assert tuple(outputs["restored"].shape) == (4, 3, 32, 32)
    assert tuple(outputs["weather_residual"].shape) == (4, 3, 32, 32)

    loss, parts = deweathering_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"reconstruction_loss", "weather_loss"}
    assert float(parts["reconstruction_loss"]) >= 0.0
    assert float(parts["weather_loss"]) >= 0.0
    loss.backward()


def test_vision_image_deweathering_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_76_synthetic_image_deweathering.data import DataConfig
    from tracks.vision.lesson_76_synthetic_image_deweathering.model import ModelConfig
    from tracks.vision.lesson_76_synthetic_image_deweathering.train import (
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
            run_name="pytest_deweathering_smoke",
            arch="deweather:deweather_cnn_tiny",
            width_mult=1.0,
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            image_size=32,
            val_fraction=0.2,
            seed=11,
            num_workers=0,
            in_channels=3,
            streak_count_min=5,
            streak_count_max=10,
            weather_strength_min=0.12,
            weather_strength_max=0.28,
        ),
        ModelConfig(
            in_channels=3,
            arch="deweather:deweather_cnn_tiny",
            width_mult=1.0,
        ),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_76_synthetic_image_deweathering" / "pytest_deweathering_smoke"
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
    for key in (
        "train_loss",
        "train_reconstruction_loss",
        "train_weather_loss",
        "eval_loss",
        "eval_reconstruction_loss",
        "eval_weather_loss",
        "eval_psnr",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
