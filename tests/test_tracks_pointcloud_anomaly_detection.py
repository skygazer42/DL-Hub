import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_pointcloud_anomaly_detection_batch_contract_and_loss_smoke() -> None:
    from tracks.pointcloud.lesson_33_toy_pointcloud_anomaly_detection.data import (
        DataConfig,
        SyntheticPointCloudAnomalyDataset,
        get_dataloaders,
    )
    from tracks.pointcloud.lesson_33_toy_pointcloud_anomaly_detection.model import (
        ModelConfig,
        anomaly_accuracy,
        anomaly_loss,
        build_model,
    )

    cfg = DataConfig(
        num_samples=32,
        num_points=48,
        batch_size=4,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        anomaly_fraction=0.4,
        anomaly_scale=0.55,
        jitter_std=0.01,
    )
    ds = SyntheticPointCloudAnomalyDataset(cfg)
    points, targets = ds[0]

    assert tuple(points.shape) == (48, 3)
    assert set(targets.keys()) == {"reconstruction", "point_labels", "label"}
    assert tuple(targets["reconstruction"].shape) == (48, 3)
    assert tuple(targets["point_labels"].shape) == (48,)
    assert tuple(targets["label"].shape) == ()
    assert points.dtype == torch.float32
    assert targets["reconstruction"].dtype == torch.float32
    assert targets["point_labels"].dtype == torch.float32
    assert targets["label"].dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    batch_points, batch_targets = next(iter(train_loader))
    assert tuple(batch_points.shape) == (4, 48, 3)
    assert tuple(batch_targets["reconstruction"].shape) == (4, 48, 3)
    assert tuple(batch_targets["point_labels"].shape) == (4, 48)
    assert tuple(batch_targets["label"].shape) == (4,)

    model = build_model(
        ModelConfig(
            in_channels=3,
            arch="recon_anomaly3d:recon_anomaly3d_tiny",
            variant="",
            width_mult=1.0,
        )
    )
    outputs = model(batch_points)
    assert set(outputs.keys()) == {"reconstruction", "point_scores", "global_score"}
    assert tuple(outputs["reconstruction"].shape) == (4, 48, 3)
    assert tuple(outputs["point_scores"].shape) == (4, 48)
    assert tuple(outputs["global_score"].shape) == (4,)

    loss, parts = anomaly_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"reconstruction_loss", "point_bce", "global_bce"}
    assert float(parts["reconstruction_loss"]) >= 0.0
    assert float(parts["point_bce"]) >= 0.0
    assert float(parts["global_bce"]) >= 0.0
    assert 0.0 <= anomaly_accuracy(outputs["global_score"], batch_targets["label"]) <= 1.0
    loss.backward()


def test_pointcloud_anomaly_detection_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.pointcloud.lesson_33_toy_pointcloud_anomaly_detection.data import DataConfig
    from tracks.pointcloud.lesson_33_toy_pointcloud_anomaly_detection.model import ModelConfig
    from tracks.pointcloud.lesson_33_toy_pointcloud_anomaly_detection.train import (
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
            run_name="pytest_pointcloud_anomaly_smoke",
            arch="recon_anomaly3d:recon_anomaly3d_tiny",
            width_mult=1.0,
        ),
        DataConfig(
            num_samples=48,
            num_points=48,
            batch_size=4,
            val_fraction=0.25,
            seed=11,
            num_workers=0,
            anomaly_fraction=0.4,
            anomaly_scale=0.55,
            jitter_std=0.01,
        ),
        ModelConfig(
            in_channels=3,
            arch="recon_anomaly3d:recon_anomaly3d_tiny",
            variant="",
            width_mult=1.0,
        ),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "pointcloud"
        / "lesson_33_toy_pointcloud_anomaly_detection"
        / "pytest_pointcloud_anomaly_smoke"
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
    for key in ("train_loss", "train_anomaly_acc", "eval_loss", "eval_anomaly_acc"):
        assert key in record
        assert float(record[key]) >= 0.0
