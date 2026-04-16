import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_pointcloud_shape_correspondence3d_batch_contract_and_loss_smoke() -> None:
    from tracks.pointcloud.lesson_35_toy_shape_correspondence_3d.data import (
        DataConfig,
        ToyShapeCorrespondenceDataset,
        get_dataloaders,
    )
    from tracks.pointcloud.lesson_35_toy_shape_correspondence_3d.model import (
        ModelConfig,
        build_model,
        correspondence_accuracy,
        correspondence_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        num_points=48,
        batch_size=4,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        translation_scale=0.2,
        noise_std=0.01,
    )

    ds = ToyShapeCorrespondenceDataset(cfg)
    source, target, correspondence = ds[0]
    assert tuple(source.shape) == (48, 3)
    assert tuple(target.shape) == (48, 3)
    assert tuple(correspondence.shape) == (48,)
    assert source.dtype == torch.float32
    assert target.dtype == torch.float32
    assert correspondence.dtype == torch.long
    assert int(correspondence.min().item()) >= 0
    assert int(correspondence.max().item()) < 48

    train_loader, _ = get_dataloaders(cfg)
    source_batch, target_batch, corr_batch = next(iter(train_loader))
    assert tuple(source_batch.shape) == (4, 48, 3)
    assert tuple(target_batch.shape) == (4, 48, 3)
    assert tuple(corr_batch.shape) == (4, 48)

    model = build_model(
        ModelConfig(
            in_channels=3,
            arch="fmnet_corr3d:fmnet_corr3d_tiny",
            variant="",
            width_mult=1.0,
        )
    )
    outputs = model(source_batch, target_batch)
    assert set(outputs.keys()) == {"scores", "matches"}
    assert tuple(outputs["scores"].shape) == (4, 48, 48)
    assert tuple(outputs["matches"].shape) == (4, 48)

    loss, parts = correspondence_loss(outputs, corr_batch)
    acc = correspondence_accuracy(outputs["matches"], corr_batch)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"cross_entropy"}
    assert float(parts["cross_entropy"]) >= 0.0
    assert 0.0 <= acc <= 1.0
    loss.backward()


def test_pointcloud_shape_correspondence3d_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.pointcloud.lesson_35_toy_shape_correspondence_3d.data import DataConfig
    from tracks.pointcloud.lesson_35_toy_shape_correspondence_3d.model import ModelConfig
    from tracks.pointcloud.lesson_35_toy_shape_correspondence_3d.train import (
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
            run_name="pytest_shape_corr3d_smoke",
            arch="fmnet_corr3d:fmnet_corr3d_tiny",
            width_mult=1.0,
        ),
        DataConfig(
            num_samples=48,
            num_points=48,
            batch_size=4,
            val_fraction=0.25,
            seed=7,
            num_workers=0,
            translation_scale=0.2,
            noise_std=0.01,
        ),
        ModelConfig(
            in_channels=3,
            arch="fmnet_corr3d:fmnet_corr3d_tiny",
            variant="",
            width_mult=1.0,
        ),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "pointcloud"
        / "lesson_35_toy_shape_correspondence_3d"
        / "pytest_shape_corr3d_smoke"
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
        value = float(record[key])
        if key.endswith("acc"):
            assert 0.0 <= value <= 1.0
        else:
            assert value >= 0.0
