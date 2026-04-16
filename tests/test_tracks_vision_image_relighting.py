import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_image_relighting_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_78_synthetic_image_relighting.data import (
        DataConfig,
        SyntheticImageRelightingDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_78_synthetic_image_relighting.model import (
        ModelConfig,
        RelightingModel,
        relighting_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        image_size=32,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=3,
        noise_std=0.01,
    )
    ds = SyntheticImageRelightingDataset(cfg)
    source, targets = ds[0]

    assert tuple(source.shape) == (3, 32, 32)
    assert set(targets.keys()) == {"relit", "light_map"}
    assert tuple(targets["relit"].shape) == (3, 32, 32)
    assert tuple(targets["light_map"].shape) == (1, 32, 32)
    assert source.dtype == torch.float32
    assert targets["relit"].dtype == torch.float32
    assert targets["light_map"].dtype == torch.float32
    assert 0.0 <= float(source.min().item()) <= float(source.max().item()) <= 1.0
    assert 0.0 <= float(targets["relit"].min().item()) <= float(targets["relit"].max().item()) <= 1.0
    assert 0.0 <= float(targets["light_map"].min().item()) <= float(targets["light_map"].max().item()) <= 1.0

    train_loader, _ = get_dataloaders(cfg)
    batch_source, batch_targets = next(iter(train_loader))
    assert tuple(batch_source.shape) == (4, 3, 32, 32)
    assert tuple(batch_targets["relit"].shape) == (4, 3, 32, 32)
    assert tuple(batch_targets["light_map"].shape) == (4, 1, 32, 32)

    model = RelightingModel(
        ModelConfig(
            in_channels=3,
            arch="deep_relight:deep_relight_tiny",
            width_mult=1.0,
        )
    )
    outputs = model(batch_source)
    assert set(outputs.keys()) == {"relit", "light_map", "residual"}
    assert tuple(outputs["relit"].shape) == (4, 3, 32, 32)
    assert tuple(outputs["light_map"].shape) == (4, 1, 32, 32)
    assert tuple(outputs["residual"].shape) == (4, 3, 32, 32)

    loss, parts = relighting_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"relit_loss", "light_map_loss"}
    assert float(parts["relit_loss"]) >= 0.0
    assert float(parts["light_map_loss"]) >= 0.0
    loss.backward()


def test_vision_image_relighting_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_78_synthetic_image_relighting.data import DataConfig
    from tracks.vision.lesson_78_synthetic_image_relighting.model import ModelConfig
    from tracks.vision.lesson_78_synthetic_image_relighting.train import (
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
            run_name="pytest_relighting_smoke",
            arch="deep_relight:deep_relight_tiny",
            width_mult=1.0,
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            image_size=32,
            val_fraction=0.2,
            seed=11,
            num_workers=0,
            in_channels=3,
            noise_std=0.01,
        ),
        ModelConfig(
            in_channels=3,
            arch="deep_relight:deep_relight_tiny",
            width_mult=1.0,
        ),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_78_synthetic_image_relighting" / "pytest_relighting_smoke"
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
        "train_relit_loss",
        "train_light_map_loss",
        "train_psnr",
        "eval_loss",
        "eval_relit_loss",
        "eval_light_map_loss",
        "eval_psnr",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
