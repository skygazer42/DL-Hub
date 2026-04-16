import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_pointcloud_gaussian_splatting_batch_contract_and_loss_smoke() -> None:
    from tracks.pointcloud.lesson_26_toy_gaussian_splatting.data import (
        DataConfig,
        ToyGaussianSplattingDataset,
        get_dataloaders,
    )
    from tracks.pointcloud.lesson_26_toy_gaussian_splatting.model import (
        ModelConfig,
        ToyGaussianSplattingModel,
        gaussian_splatting_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        num_points=48,
        image_size=24,
        batch_size=4,
        val_fraction=0.25,
        seed=0,
        num_workers=0,
        noise_std=0.03,
        p_sphere=0.5,
        splat_sigma=0.08,
    )

    ds = ToyGaussianSplattingDataset(cfg)
    observed, target = ds[0]
    assert tuple(observed.shape) == (48, 3)
    assert tuple(target.shape) == (1, 24, 24)
    assert observed.dtype == torch.float32
    assert target.dtype == torch.float32
    assert bool(torch.isfinite(observed).all())
    assert bool(torch.isfinite(target).all())

    train_loader, _ = get_dataloaders(cfg)
    observed_batch, target_batch = next(iter(train_loader))
    assert tuple(observed_batch.shape) == (4, 48, 3)
    assert tuple(target_batch.shape) == (4, 1, 24, 24)

    model = ToyGaussianSplattingModel(
        ModelConfig(
            hidden_features=32,
            image_size=24,
            min_sigma=0.02,
        )
    )
    pred = model(observed_batch)
    assert tuple(pred.shape) == (4, 1, 24, 24)

    loss, parts = gaussian_splatting_loss(pred, target_batch)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"mse", "mass_l1"}
    assert float(parts["mse"]) >= 0.0
    assert float(parts["mass_l1"]) >= 0.0
    loss.backward()


def test_pointcloud_gaussian_splatting_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.pointcloud.lesson_26_toy_gaussian_splatting.data import DataConfig
    from tracks.pointcloud.lesson_26_toy_gaussian_splatting.model import ModelConfig
    from tracks.pointcloud.lesson_26_toy_gaussian_splatting.train import (
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
            run_name="pytest_gaussian_splatting_smoke",
        ),
        DataConfig(
            num_samples=40,
            num_points=48,
            image_size=24,
            batch_size=4,
            val_fraction=0.25,
            seed=7,
            num_workers=0,
            noise_std=0.03,
            p_sphere=0.5,
            splat_sigma=0.08,
        ),
        ModelConfig(
            hidden_features=32,
            image_size=24,
            min_sigma=0.02,
        ),
    )

    assert exit_code == 0

    run_dir = (
        tmp_path
        / "pointcloud"
        / "lesson_26_toy_gaussian_splatting"
        / "pytest_gaussian_splatting_smoke"
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
    for key in ("train_loss", "train_mse", "eval_loss", "eval_mse"):
        assert key in record
        assert float(record[key]) >= 0.0
