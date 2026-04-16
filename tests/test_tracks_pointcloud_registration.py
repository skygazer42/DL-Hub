import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_pointcloud_registration_batch_contract_and_loss_smoke() -> None:
    from tracks.pointcloud.lesson_36_toy_pointcloud_registration.data import (
        DataConfig,
        ToyPointCloudRegistrationDataset,
        get_dataloaders,
    )
    from tracks.pointcloud.lesson_36_toy_pointcloud_registration.model import (
        ModelConfig,
        build_model,
        pose_l1_error,
        registration_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        num_points=48,
        batch_size=4,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        translation_scale=0.3,
        rotation_scale=0.4,
        noise_std=0.01,
    )

    ds = ToyPointCloudRegistrationDataset(cfg)
    source, target, pose6d = ds[0]
    assert tuple(source.shape) == (48, 3)
    assert tuple(target.shape) == (48, 3)
    assert tuple(pose6d.shape) == (6,)
    assert source.dtype == torch.float32
    assert target.dtype == torch.float32
    assert pose6d.dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    source_batch, target_batch, pose_batch = next(iter(train_loader))
    assert tuple(source_batch.shape) == (4, 48, 3)
    assert tuple(target_batch.shape) == (4, 48, 3)
    assert tuple(pose_batch.shape) == (4, 6)

    model = build_model(
        ModelConfig(
            in_channels=3,
            arch="pointnetlk:pointnetlk_tiny",
            variant="",
            width_mult=1.0,
        )
    )
    outputs = model(source_batch, target_batch)
    assert set(outputs.keys()) == {"pose6d"}
    assert tuple(outputs["pose6d"].shape) == (4, 6)

    loss, parts = registration_loss(outputs, pose_batch)
    err = pose_l1_error(outputs["pose6d"], pose_batch)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"translation_mse", "rotation_mse"}
    assert float(parts["translation_mse"]) >= 0.0
    assert float(parts["rotation_mse"]) >= 0.0
    assert err >= 0.0
    loss.backward()


def test_pointcloud_registration_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.pointcloud.lesson_36_toy_pointcloud_registration.data import DataConfig
    from tracks.pointcloud.lesson_36_toy_pointcloud_registration.model import ModelConfig
    from tracks.pointcloud.lesson_36_toy_pointcloud_registration.train import (
        TrainConfig,
        run_training,
    )

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=5e-3,
            seed=0,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_pointcloud_registration_smoke",
            arch="pointnetlk:pointnetlk_tiny",
            width_mult=1.0,
        ),
        DataConfig(
            num_samples=48,
            num_points=48,
            batch_size=4,
            val_fraction=0.25,
            seed=7,
            num_workers=0,
            translation_scale=0.3,
            rotation_scale=0.4,
            noise_std=0.01,
        ),
        ModelConfig(
            in_channels=3,
            arch="pointnetlk:pointnetlk_tiny",
            variant="",
            width_mult=1.0,
        ),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "pointcloud"
        / "lesson_36_toy_pointcloud_registration"
        / "pytest_pointcloud_registration_smoke"
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
    for key in ("train_loss", "train_pose_l1", "eval_loss", "eval_pose_l1"):
        assert key in record
        assert float(record[key]) >= 0.0
