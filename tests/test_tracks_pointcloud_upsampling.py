import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_pointcloud_upsampling_batch_contract_and_loss_smoke() -> None:
    from dlhub.pointcloud.ops import chamfer_distance
    from tracks.pointcloud.lesson_34_toy_pointcloud_upsampling.data import (
        DataConfig,
        ToyPointCloudUpsamplingDataset,
        get_dataloaders,
    )
    from tracks.pointcloud.lesson_34_toy_pointcloud_upsampling.model import (
        ModelConfig,
        build_model,
    )

    cfg = DataConfig(
        num_samples=32,
        num_sparse_points=32,
        upsample_factor=2,
        batch_size=4,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        p_sphere=0.5,
    )
    dataset = ToyPointCloudUpsamplingDataset(cfg)
    sparse, dense = dataset[0]
    assert sparse.shape == (32, 3)
    assert dense.shape == (64, 3)

    train_loader, _ = get_dataloaders(cfg)
    sparse_batch, dense_batch = next(iter(train_loader))
    assert sparse_batch.shape == (4, 32, 3)
    assert dense_batch.shape == (4, 64, 3)

    model = build_model(
        ModelConfig(
            in_channels=3,
            arch="punet_upsample:punet_upsample_tiny",
            variant="",
            width_mult=1.0,
        )
    )
    pred = model(sparse_batch)
    assert pred.shape == (4, 64, 3)
    loss = chamfer_distance(pred, dense_batch)
    assert torch.isfinite(loss)
    loss.backward()


def test_pointcloud_upsampling_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.pointcloud.lesson_34_toy_pointcloud_upsampling.data import DataConfig
    from tracks.pointcloud.lesson_34_toy_pointcloud_upsampling.train import (
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
            run_name="pytest_pointcloud_upsampling_smoke",
            arch="punet_upsample:punet_upsample_tiny",
            width_mult=1.0,
        ),
        DataConfig(
            num_samples=32,
            num_sparse_points=32,
            upsample_factor=2,
            batch_size=4,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            p_sphere=0.5,
        ),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "pointcloud"
        / "lesson_34_toy_pointcloud_upsampling"
        / "pytest_pointcloud_upsampling_smoke"
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
