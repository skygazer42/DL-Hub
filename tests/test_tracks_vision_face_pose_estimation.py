import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_face_pose_estimation_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_47_synthetic_face_pose_estimation.data import (
        DataConfig,
        SyntheticFacePoseDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_47_synthetic_face_pose_estimation.model import (
        FacePoseRegressor,
        ModelConfig,
        pose_loss,
        pose_mae,
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
    ds = SyntheticFacePoseDataset(cfg)
    image, pose = ds[0]
    assert tuple(image.shape) == (1, 48, 48)
    assert tuple(pose.shape) == (3,)
    assert torch.all(pose >= -1.0)
    assert torch.all(pose <= 1.0)

    train_loader, _ = get_dataloaders(cfg)
    images, poses = next(iter(train_loader))
    assert tuple(images.shape) == (5, 1, 48, 48)
    assert tuple(poses.shape) == (5, 3)

    model = FacePoseRegressor(ModelConfig(in_channels=1, hidden_channels=24, num_blocks=3, dropout=0.0))
    pred = model(images)
    assert tuple(pred.shape) == (5, 3)

    loss = pose_loss(pred, poses)
    assert torch.isfinite(loss)
    assert pose_mae(pred.detach(), poses) >= 0.0
    loss.backward()


def test_face_pose_estimation_training_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.vision.lesson_47_synthetic_face_pose_estimation.data import DataConfig
    from tracks.vision.lesson_47_synthetic_face_pose_estimation.model import ModelConfig
    from tracks.vision.lesson_47_synthetic_face_pose_estimation.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_face_pose_estimation_smoke",
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

    run_dir = tmp_path / "vision" / "lesson_47_synthetic_face_pose_estimation" / "pytest_face_pose_estimation_smoke"
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
