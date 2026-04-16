import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_event_camera_understanding_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_80_synthetic_event_camera_understanding.data import (
        DataConfig,
        SyntheticEventCameraDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_80_synthetic_event_camera_understanding.model import (
        EventUnderstandingModel,
        ModelConfig,
        event_understanding_loss,
    )

    cfg = DataConfig(
        num_samples=40,
        batch_size=4,
        image_size=24,
        num_bins=5,
        val_fraction=0.25,
        seed=9,
        num_workers=0,
        noise_std=0.01,
    )
    ds = SyntheticEventCameraDataset(cfg)
    event_volume, target = ds[0]

    assert tuple(event_volume.shape) == (5, 24, 24)
    assert set(target.keys()) == {"polarity_map", "motion", "depth_like"}
    assert tuple(target["polarity_map"].shape) == (2, 24, 24)
    assert tuple(target["motion"].shape) == (2, 24, 24)
    assert tuple(target["depth_like"].shape) == (1, 24, 24)
    assert event_volume.dtype == torch.float32
    assert target["polarity_map"].dtype == torch.float32
    assert target["motion"].dtype == torch.float32
    assert target["depth_like"].dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    batch_events, batch_targets = next(iter(train_loader))
    assert tuple(batch_events.shape) == (4, 5, 24, 24)
    assert tuple(batch_targets["polarity_map"].shape) == (4, 2, 24, 24)
    assert tuple(batch_targets["motion"].shape) == (4, 2, 24, 24)
    assert tuple(batch_targets["depth_like"].shape) == (4, 1, 24, 24)

    model = EventUnderstandingModel(
        ModelConfig(
            in_channels=5,
            hidden_channels=16,
            family="ev_cnn",
            variant="ev_cnn_tiny",
        )
    )
    outputs = model(batch_events)
    assert set(outputs.keys()) == {"polarity_map", "motion", "depth_like"}
    assert tuple(outputs["polarity_map"].shape) == (4, 2, 24, 24)
    assert tuple(outputs["motion"].shape) == (4, 2, 24, 24)
    assert tuple(outputs["depth_like"].shape) == (4, 1, 24, 24)

    loss, parts = event_understanding_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"polarity_loss", "motion_loss", "depth_loss"}
    assert float(parts["polarity_loss"]) >= 0.0
    assert float(parts["motion_loss"]) >= 0.0
    assert float(parts["depth_loss"]) >= 0.0
    loss.backward()


def test_vision_event_camera_understanding_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_80_synthetic_event_camera_understanding.data import DataConfig
    from tracks.vision.lesson_80_synthetic_event_camera_understanding.model import ModelConfig
    from tracks.vision.lesson_80_synthetic_event_camera_understanding.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=80,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_event_camera_smoke",
        ),
        DataConfig(
            num_samples=64,
            batch_size=4,
            image_size=24,
            num_bins=5,
            val_fraction=0.25,
            seed=19,
            num_workers=0,
            noise_std=0.01,
        ),
        ModelConfig(
            in_channels=5,
            hidden_channels=16,
            family="ev_cnn",
            variant="ev_cnn_tiny",
        ),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "vision"
        / "lesson_80_synthetic_event_camera_understanding"
        / "pytest_event_camera_smoke"
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
    for key in (
        "train_loss",
        "train_polarity_loss",
        "train_motion_loss",
        "train_depth_loss",
        "eval_loss",
        "eval_polarity_loss",
        "eval_motion_loss",
        "eval_depth_loss",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
