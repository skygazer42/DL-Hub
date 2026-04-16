import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_finger_spread_estimation_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_56_synthetic_finger_spread_estimation.data import (
        DataConfig,
        SyntheticFingerSpreadDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_56_synthetic_finger_spread_estimation.model import (
        FingerSpreadRegressor,
        ModelConfig,
        finger_spread_loss,
        finger_spread_mae,
    )

    cfg = DataConfig(
        num_samples=40,
        batch_size=5,
        image_size=64,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        in_channels=1,
    )

    dataset = SyntheticFingerSpreadDataset(cfg)
    image, target = dataset[0]
    assert tuple(image.shape) == (1, 64, 64)
    assert tuple(target.shape) == (1,)
    assert 0.0 <= float(target.item()) <= 1.0
    assert image.dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    images, targets = next(iter(train_loader))
    assert tuple(images.shape) == (5, 1, 64, 64)
    assert tuple(targets.shape) == (5, 1)
    assert targets.dtype == torch.float32

    model = FingerSpreadRegressor(ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, dropout=0.0))
    predictions = model(images)
    assert tuple(predictions.shape) == (5, 1)

    loss = finger_spread_loss(predictions, targets)
    assert torch.isfinite(loss)
    assert finger_spread_mae(predictions.detach(), targets) >= 0.0
    loss.backward()


def test_finger_spread_estimation_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_56_synthetic_finger_spread_estimation.data import DataConfig
    from tracks.vision.lesson_56_synthetic_finger_spread_estimation.model import ModelConfig
    from tracks.vision.lesson_56_synthetic_finger_spread_estimation.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=56,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_finger_spread_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=6,
            image_size=64,
            val_fraction=0.2,
            seed=7,
            num_workers=0,
            in_channels=1,
        ),
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, dropout=0.0),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "vision"
        / "lesson_56_synthetic_finger_spread_estimation"
        / "pytest_finger_spread_smoke"
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
