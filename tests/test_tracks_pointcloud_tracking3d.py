import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_pointcloud_tracking3d_batch_contract_and_loss_smoke() -> None:
    from tracks.pointcloud.lesson_30_compact_3d_object_tracking.data import (
        DataConfig,
        SyntheticObjectTrackingDataset,
        get_dataloaders,
    )
    from tracks.pointcloud.lesson_30_compact_3d_object_tracking.model import (
        ModelConfig,
        CompactObjectTracker,
        tracking_loss,
    )

    cfg = DataConfig(
        num_samples=40,
        num_points=64,
        batch_size=5,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        motion_scale=0.25,
        clutter_ratio=0.25,
        noise_std=0.01,
    )
    ds = SyntheticObjectTrackingDataset(cfg)
    prev_cloud, curr_cloud, target_state = ds[0]
    assert tuple(prev_cloud.shape) == (64, 3)
    assert tuple(curr_cloud.shape) == (64, 3)
    assert tuple(target_state.shape) == (6,)
    assert prev_cloud.dtype == torch.float32
    assert curr_cloud.dtype == torch.float32
    assert target_state.dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    prev_batch, curr_batch, target_batch = next(iter(train_loader))
    assert tuple(prev_batch.shape) == (5, 64, 3)
    assert tuple(curr_batch.shape) == (5, 64, 3)
    assert tuple(target_batch.shape) == (5, 6)

    model = CompactObjectTracker(ModelConfig(hidden_features=32))
    pred_state = model(prev_batch, curr_batch)
    assert tuple(pred_state.shape) == (5, 6)

    loss, parts = tracking_loss(pred_state, target_batch)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"state_mse", "center_mae", "velocity_mae"}
    for key in ("state_mse", "center_mae", "velocity_mae"):
        assert float(parts[key]) >= 0.0
    loss.backward()


def test_pointcloud_tracking3d_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.pointcloud.lesson_30_compact_3d_object_tracking.data import DataConfig
    from tracks.pointcloud.lesson_30_compact_3d_object_tracking.model import ModelConfig
    from tracks.pointcloud.lesson_30_compact_3d_object_tracking.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=5e-3,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_tracking3d_smoke",
        ),
        DataConfig(
            num_samples=40,
            num_points=64,
            batch_size=5,
            val_fraction=0.25,
            seed=7,
            num_workers=0,
            motion_scale=0.25,
            clutter_ratio=0.25,
            noise_std=0.01,
        ),
        ModelConfig(hidden_features=32),
    )
    assert exit_code == 0

    run_dir = tmp_path / "pointcloud" / "lesson_30_compact_3d_object_tracking" / "pytest_tracking3d_smoke"
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
        "train_center_mae",
        "train_velocity_mae",
        "eval_loss",
        "eval_center_mae",
        "eval_velocity_mae",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
