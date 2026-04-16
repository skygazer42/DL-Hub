import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_video_restoration_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_69_synthetic_video_restoration.data import (
        DataConfig,
        SyntheticVideoRestorationDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_69_synthetic_video_restoration.model import (
        ModelConfig,
        VideoRestorationModel,
        restoration_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        frames=5,
        image_size=32,
        val_fraction=0.25,
        seed=5,
        num_workers=0,
        in_channels=1,
        noise_std=0.08,
    )
    ds = SyntheticVideoRestorationDataset(cfg)
    degraded, clean = ds[0]
    assert tuple(degraded.shape) == (5, 1, 32, 32)
    assert tuple(clean.shape) == (5, 1, 32, 32)
    assert degraded.dtype == torch.float32
    assert clean.dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    degraded_batch, clean_batch = next(iter(train_loader))
    assert tuple(degraded_batch.shape) == (4, 5, 1, 32, 32)
    assert tuple(clean_batch.shape) == (4, 5, 1, 32, 32)

    model = VideoRestorationModel(ModelConfig(in_channels=1, hidden_channels=16, num_blocks=3))
    outputs = model(degraded_batch)
    assert set(outputs.keys()) == {"restored"}
    assert tuple(outputs["restored"].shape) == (4, 5, 1, 32, 32)

    loss, parts = restoration_loss(outputs["restored"], clean_batch)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"l1_loss", "temporal_loss"}
    assert float(parts["l1_loss"]) >= 0.0
    assert float(parts["temporal_loss"]) >= 0.0
    loss.backward()


def test_vision_video_restoration_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_69_synthetic_video_restoration.data import DataConfig
    from tracks.vision.lesson_69_synthetic_video_restoration.model import ModelConfig
    from tracks.vision.lesson_69_synthetic_video_restoration.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=11,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_video_restoration_smoke",
        ),
        DataConfig(
            num_samples=40,
            batch_size=4,
            frames=5,
            image_size=32,
            val_fraction=0.25,
            seed=7,
            num_workers=0,
            in_channels=1,
            noise_std=0.08,
        ),
        ModelConfig(in_channels=1, hidden_channels=16, num_blocks=3),
    )
    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_69_synthetic_video_restoration" / "pytest_video_restoration_smoke"
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
        "train_l1_loss",
        "train_temporal_loss",
        "eval_loss",
        "eval_l1_loss",
        "eval_temporal_loss",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
