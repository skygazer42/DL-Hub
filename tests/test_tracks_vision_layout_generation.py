import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_layout_generation_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_82_synthetic_layout_generation.data import (
        DataConfig,
        SyntheticLayoutGenerationDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_82_synthetic_layout_generation.model import (
        LayoutGenerationModel,
        ModelConfig,
        layout_generation_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        image_size=32,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=3,
        max_objects=3,
        noise_std=0.01,
    )
    ds = SyntheticLayoutGenerationDataset(cfg)
    condition, targets = ds[0]

    assert tuple(condition.shape) == (3, 32, 32)
    assert set(targets.keys()) == {"layout", "occupancy"}
    assert tuple(targets["layout"].shape) == (3, 32, 32)
    assert tuple(targets["occupancy"].shape) == (1, 32, 32)
    assert condition.dtype == torch.float32
    assert targets["layout"].dtype == torch.float32
    assert targets["occupancy"].dtype == torch.float32
    assert 0.0 <= float(condition.min().item()) <= float(condition.max().item()) <= 1.0
    assert 0.0 <= float(targets["layout"].min().item()) <= float(targets["layout"].max().item()) <= 1.0
    assert 0.0 <= float(targets["occupancy"].min().item()) <= float(targets["occupancy"].max().item()) <= 1.0

    train_loader, _ = get_dataloaders(cfg)
    batch_condition, batch_targets = next(iter(train_loader))
    assert tuple(batch_condition.shape) == (4, 3, 32, 32)
    assert tuple(batch_targets["layout"].shape) == (4, 3, 32, 32)
    assert tuple(batch_targets["occupancy"].shape) == (4, 1, 32, 32)

    model = LayoutGenerationModel(
        ModelConfig(
            in_channels=3,
            hidden_channels=24,
            family="layouttransformer",
            variant="layouttransformer_tiny",
            width_mult=1.0,
        )
    )
    outputs = model(batch_condition)
    assert set(outputs.keys()) == {"layout", "occupancy", "residual"}
    assert tuple(outputs["layout"].shape) == (4, 3, 32, 32)
    assert tuple(outputs["occupancy"].shape) == (4, 1, 32, 32)
    assert tuple(outputs["residual"].shape) == (4, 3, 32, 32)

    loss, parts = layout_generation_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"layout_loss", "occupancy_loss"}
    assert float(parts["layout_loss"]) >= 0.0
    assert float(parts["occupancy_loss"]) >= 0.0
    loss.backward()


def test_vision_layout_generation_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_82_synthetic_layout_generation.data import DataConfig
    from tracks.vision.lesson_82_synthetic_layout_generation.model import ModelConfig
    from tracks.vision.lesson_82_synthetic_layout_generation.train import (
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
            run_name="pytest_layout_generation_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            image_size=32,
            val_fraction=0.2,
            seed=11,
            num_workers=0,
            in_channels=3,
            max_objects=3,
            noise_std=0.01,
        ),
        ModelConfig(
            in_channels=3,
            hidden_channels=24,
            family="layouttransformer",
            variant="layouttransformer_tiny",
            width_mult=1.0,
        ),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_82_synthetic_layout_generation" / "pytest_layout_generation_smoke"
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
        "train_layout_loss",
        "train_occupancy_loss",
        "train_psnr",
        "eval_loss",
        "eval_layout_loss",
        "eval_occupancy_loss",
        "eval_psnr",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
