import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_pointcloud_segmentation3d_batch_contract_and_loss_smoke() -> None:
    from tracks.pointcloud.lesson_28_toy_3d_semantic_segmentation.data import (
        DataConfig,
        ToySemanticSegmentation3DDataset,
        get_dataloaders,
    )
    from tracks.pointcloud.lesson_28_toy_3d_semantic_segmentation.model import (
        ModelConfig,
        ToyPointNetSemanticSeg3D,
        segmentation3d_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        num_points=96,
        num_classes=4,
        batch_size=4,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        jitter_std=0.01,
    )

    ds = ToySemanticSegmentation3DDataset(cfg)
    points, labels = ds[0]
    assert tuple(points.shape) == (96, 3)
    assert tuple(labels.shape) == (96,)
    assert points.dtype == torch.float32
    assert labels.dtype == torch.long
    assert int(labels.min().item()) >= 0
    assert int(labels.max().item()) < 4

    train_loader, _ = get_dataloaders(cfg)
    points_batch, labels_batch = next(iter(train_loader))
    assert tuple(points_batch.shape) == (4, 96, 3)
    assert tuple(labels_batch.shape) == (4, 96)

    model = ToyPointNetSemanticSeg3D(
        ModelConfig(in_channels=3, hidden_features=32, num_classes=4, dropout=0.0)
    )
    logits = model(points_batch)
    assert tuple(logits.shape) == (4, 96, 4)

    loss, stats = segmentation3d_loss(logits, labels_batch)
    assert torch.isfinite(loss)
    assert set(stats.keys()) == {"loss_ce", "acc"}
    assert float(stats["loss_ce"]) >= 0.0
    assert 0.0 <= float(stats["acc"]) <= 1.0
    loss.backward()


def test_pointcloud_segmentation3d_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.pointcloud.lesson_28_toy_3d_semantic_segmentation.data import DataConfig
    from tracks.pointcloud.lesson_28_toy_3d_semantic_segmentation.model import ModelConfig
    from tracks.pointcloud.lesson_28_toy_3d_semantic_segmentation.train import (
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
            run_name="pytest_pointcloud_seg3d_smoke",
        ),
        DataConfig(
            num_samples=48,
            num_points=96,
            num_classes=4,
            batch_size=4,
            val_fraction=0.25,
            seed=9,
            num_workers=0,
            jitter_std=0.01,
        ),
        ModelConfig(in_channels=3, hidden_features=32, num_classes=4, dropout=0.0),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "pointcloud"
        / "lesson_28_toy_3d_semantic_segmentation"
        / "pytest_pointcloud_seg3d_smoke"
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
    assert float(record["eval_loss"]) >= 0.0
    assert 0.0 <= float(record["train_acc"]) <= 1.0
    assert 0.0 <= float(record["eval_acc"]) <= 1.0
