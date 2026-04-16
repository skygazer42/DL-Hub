import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_gaze_estimation_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_48_synthetic_gaze_estimation.data import (
        DataConfig,
        SyntheticGazeDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_48_synthetic_gaze_estimation.model import (
        GazeRegressor,
        ModelConfig,
        gaze_l1,
        gaze_loss,
    )

    cfg = DataConfig(
        num_samples=40,
        batch_size=5,
        image_size=48,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=1,
    )
    ds = SyntheticGazeDataset(cfg)
    image, target = ds[0]
    assert tuple(image.shape) == (1, 48, 48)
    assert tuple(target.shape) == (2,)
    assert torch.all(target >= 0.0)
    assert torch.all(target <= 1.0)

    train_loader, _ = get_dataloaders(cfg)
    images, targets = next(iter(train_loader))
    assert tuple(images.shape) == (5, 1, 48, 48)
    assert tuple(targets.shape) == (5, 2)

    model = GazeRegressor(ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, dropout=0.0))
    pred = model(images)
    assert tuple(pred.shape) == (5, 2)

    loss = gaze_loss(pred, targets)
    assert torch.isfinite(loss)
    assert gaze_l1(pred.detach(), targets) >= 0.0
    loss.backward()


def test_gaze_estimation_training_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.vision.lesson_48_synthetic_gaze_estimation.data import DataConfig
    from tracks.vision.lesson_48_synthetic_gaze_estimation.model import ModelConfig
    from tracks.vision.lesson_48_synthetic_gaze_estimation.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=48,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_gaze_estimation_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=6,
            image_size=48,
            val_fraction=0.2,
            seed=5,
            num_workers=0,
            in_channels=1,
        ),
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, dropout=0.1),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_48_synthetic_gaze_estimation" / "pytest_gaze_estimation_smoke"
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
    for key in ("train_loss", "train_l1", "eval_loss", "eval_l1"):
        assert key in record
        assert float(record[key]) >= 0.0
