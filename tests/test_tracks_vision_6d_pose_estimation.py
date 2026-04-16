import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_six_d_pose_estimation_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_35_synthetic_6d_pose_estimation.data import (
        DataConfig,
        SyntheticPoseDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_35_synthetic_6d_pose_estimation.model import (
        ModelConfig,
        PoseRegressor,
        mean_pose_mae,
        pose_regression_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        image_size=48,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=1,
        pose_dim=9,
    )
    dataset = SyntheticPoseDataset(cfg)
    image, pose = dataset[0]
    assert tuple(image.shape) == (1, 48, 48)
    assert tuple(pose.shape) == (9,)
    assert torch.isfinite(pose).all()

    train_loader, _ = get_dataloaders(cfg)
    images, pose_targets = next(iter(train_loader))
    assert tuple(images.shape) == (4, 1, 48, 48)
    assert tuple(pose_targets.shape) == (4, 9)

    model = PoseRegressor(
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, pose_dim=9, dropout=0.0)
    )
    predictions = model(images)
    assert tuple(predictions.shape) == (4, 9)

    loss = pose_regression_loss(predictions, pose_targets)
    assert torch.isfinite(loss)
    assert mean_pose_mae(predictions.detach(), pose_targets) >= 0.0
    loss.backward()


def test_six_d_pose_estimation_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_35_synthetic_6d_pose_estimation.data import DataConfig
    from tracks.vision.lesson_35_synthetic_6d_pose_estimation.model import ModelConfig
    from tracks.vision.lesson_35_synthetic_6d_pose_estimation.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_pose6d_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            image_size=48,
            val_fraction=0.2,
            seed=5,
            num_workers=0,
            in_channels=1,
            pose_dim=9,
        ),
        ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, pose_dim=9, dropout=0.1),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_35_synthetic_6d_pose_estimation" / "pytest_pose6d_smoke"
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
    for key in ("train_loss", "eval_loss", "eval_pose_mae"):
        assert key in record
        assert float(record[key]) >= 0.0
