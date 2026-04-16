import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_pointcloud_instance_segmentation3d_batch_contract_and_loss_smoke() -> None:
    from tracks.pointcloud.lesson_29_toy_3d_instance_segmentation.data import (
        DataConfig,
        ToyInstanceSegmentation3DDataset,
        get_dataloaders,
    )
    from tracks.pointcloud.lesson_29_toy_3d_instance_segmentation.model import (
        ModelConfig,
        ToyInstanceSegmentation3DNet,
        instance_segmentation_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        num_points=48,
        batch_size=4,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        cluster_offset=0.9,
        cluster_std=0.08,
    )

    ds = ToyInstanceSegmentation3DDataset(cfg)
    points, instance_ids = ds[0]
    assert tuple(points.shape) == (48, 3)
    assert tuple(instance_ids.shape) == (48,)
    assert points.dtype == torch.float32
    assert instance_ids.dtype == torch.long
    assert set(instance_ids.unique().tolist()) <= {0, 1}

    train_loader, _ = get_dataloaders(cfg)
    points_batch, ids_batch = next(iter(train_loader))
    assert tuple(points_batch.shape) == (4, 48, 3)
    assert tuple(ids_batch.shape) == (4, 48)

    model = ToyInstanceSegmentation3DNet(ModelConfig(hidden_features=32, embedding_dim=8))
    pred = model(points_batch)
    assert set(pred.keys()) == {"embeddings", "logits"}
    assert tuple(pred["embeddings"].shape) == (4, 48, 8)
    assert tuple(pred["logits"].shape) == (4, 48, 2)

    loss, parts = instance_segmentation_loss(pred["logits"], ids_batch)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"ce_loss", "point_acc"}
    assert float(parts["ce_loss"]) >= 0.0
    assert 0.0 <= float(parts["point_acc"]) <= 1.0
    loss.backward()


def test_pointcloud_instance_segmentation3d_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.pointcloud.lesson_29_toy_3d_instance_segmentation.data import DataConfig
    from tracks.pointcloud.lesson_29_toy_3d_instance_segmentation.model import ModelConfig
    from tracks.pointcloud.lesson_29_toy_3d_instance_segmentation.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=3e-3,
            seed=0,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_instance_seg3d_smoke",
        ),
        DataConfig(
            num_samples=40,
            num_points=48,
            batch_size=4,
            val_fraction=0.25,
            seed=7,
            num_workers=0,
            cluster_offset=0.9,
            cluster_std=0.08,
        ),
        ModelConfig(hidden_features=32, embedding_dim=8),
    )
    assert exit_code == 0

    run_dir = (
        tmp_path
        / "pointcloud"
        / "lesson_29_toy_3d_instance_segmentation"
        / "pytest_instance_seg3d_smoke"
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
    for key in ("train_loss", "train_acc", "eval_loss", "eval_acc"):
        assert key in record
    assert float(record["train_loss"]) >= 0.0
    assert 0.0 <= float(record["train_acc"]) <= 1.0
    assert float(record["eval_loss"]) >= 0.0
    assert 0.0 <= float(record["eval_acc"]) <= 1.0
