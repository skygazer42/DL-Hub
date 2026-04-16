import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_pointcloud_completion_batch_contract_and_loss_smoke() -> None:
    from dlhub.pointcloud.ops import chamfer_distance
    from tracks.pointcloud.lesson_24_toy_pointcloud_completion.data import (
        DataConfig,
        ToyPointCloudCompletionDataset,
        get_dataloaders,
    )
    from tracks.pointcloud.lesson_24_toy_pointcloud_completion.model import (
        ModelConfig,
        build_model,
    )

    cfg = DataConfig(
        num_samples=32,
        num_points=48,
        batch_size=4,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        visible_fraction=0.55,
        p_sphere=0.5,
    )
    dataset = ToyPointCloudCompletionDataset(cfg)
    partial, complete = dataset[0]
    assert partial.shape == (48, 3)
    assert complete.shape == (48, 3)

    train_loader, _ = get_dataloaders(cfg)
    partial_batch, complete_batch = next(iter(train_loader))
    assert partial_batch.shape == (4, 48, 3)
    assert complete_batch.shape == (4, 48, 3)

    model = build_model(
        ModelConfig(
            in_channels=3,
            num_points=48,
            arch="pointnet_ae:pointnet_ae_tiny",
            variant="",
            dropout=0.0,
        )
    )
    pred = model(partial_batch)
    assert pred.shape == (4, 48, 3)
    loss = chamfer_distance(pred, complete_batch)
    assert torch.isfinite(loss)
    loss.backward()


def test_pointcloud_completion_training_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.pointcloud.lesson_24_toy_pointcloud_completion.data import DataConfig
    from tracks.pointcloud.lesson_24_toy_pointcloud_completion.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=2e-3,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_pointcloud_completion_smoke",
            arch="pointnet_ae:pointnet_ae_tiny",
            dropout=0.0,
        ),
        DataConfig(
            num_samples=32,
            num_points=48,
            batch_size=4,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            visible_fraction=0.55,
            p_sphere=0.5,
        ),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "pointcloud"
        / "lesson_24_toy_pointcloud_completion"
        / "pytest_pointcloud_completion_smoke"
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
    for key in ("train_chamfer", "eval_chamfer"):
        assert key in record
        assert float(record[key]) >= 0.0
