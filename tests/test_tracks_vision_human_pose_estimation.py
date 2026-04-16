import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_human_pose_estimation_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_49_synthetic_human_pose_estimation.data import (
        DataConfig,
        SyntheticHumanPoseDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_49_synthetic_human_pose_estimation.model import (
        HumanPoseRegressor,
        ModelConfig,
        human_pose_loss,
        mean_pose_l2_pixels,
    )

    cfg = DataConfig(
        num_samples=36,
        batch_size=4,
        image_size=64,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        in_channels=1,
        num_keypoints=8,
    )

    ds = SyntheticHumanPoseDataset(cfg)
    image, keypoints = ds[0]
    assert tuple(image.shape) == (1, 64, 64)
    assert tuple(keypoints.shape) == (16,)
    assert image.dtype == torch.float32
    assert keypoints.dtype == torch.float32
    assert torch.all(keypoints >= 0.0)
    assert torch.all(keypoints <= 1.0)

    train_loader, _ = get_dataloaders(cfg)
    images, target_keypoints = next(iter(train_loader))
    assert tuple(images.shape) == (4, 1, 64, 64)
    assert tuple(target_keypoints.shape) == (4, 16)

    model = HumanPoseRegressor(
        ModelConfig(
            in_channels=1,
            hidden_channels=24,
            num_blocks=3,
            num_keypoints=8,
            dropout=0.0,
        )
    )
    predictions = model(images)
    assert tuple(predictions.shape) == (4, 16)

    loss = human_pose_loss(predictions, target_keypoints)
    assert torch.isfinite(loss)
    assert mean_pose_l2_pixels(predictions.detach(), target_keypoints, image_size=64) >= 0.0
    loss.backward()


def test_human_pose_estimation_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_49_synthetic_human_pose_estimation.data import DataConfig
    from tracks.vision.lesson_49_synthetic_human_pose_estimation.model import ModelConfig
    from tracks.vision.lesson_49_synthetic_human_pose_estimation.train import (
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
            run_name="pytest_human_pose_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=6,
            image_size=64,
            val_fraction=0.25,
            seed=7,
            num_workers=0,
            in_channels=1,
            num_keypoints=8,
        ),
        ModelConfig(
            in_channels=1,
            hidden_channels=24,
            num_blocks=3,
            num_keypoints=8,
            dropout=0.1,
        ),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "vision"
        / "lesson_49_synthetic_human_pose_estimation"
        / "pytest_human_pose_smoke"
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
    for key in ("train_mse", "eval_mse", "eval_l2_px"):
        assert key in record
        assert float(record[key]) >= 0.0
