import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_image_deraining_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_60_synthetic_image_deraining.data import (
        DataConfig,
        SyntheticImageDerainingDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_60_synthetic_image_deraining.model import (
        DerainingModel,
        ModelConfig,
        deraining_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        image_size=32,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=3,
        rain_lines_min=6,
        rain_lines_max=12,
        rain_strength_min=0.12,
        rain_strength_max=0.28,
    )
    ds = SyntheticImageDerainingDataset(cfg)
    rainy, targets = ds[0]

    assert tuple(rainy.shape) == (3, 32, 32)
    assert set(targets.keys()) == {"clean", "rain_layer"}
    assert tuple(targets["clean"].shape) == (3, 32, 32)
    assert tuple(targets["rain_layer"].shape) == (3, 32, 32)
    assert rainy.dtype == torch.float32
    assert targets["clean"].dtype == torch.float32
    assert targets["rain_layer"].dtype == torch.float32
    assert 0.0 <= float(rainy.min().item()) <= float(rainy.max().item()) <= 1.0
    assert 0.0 <= float(targets["rain_layer"].min().item()) <= float(
        targets["rain_layer"].max().item()
    ) <= 1.0

    train_loader, _ = get_dataloaders(cfg)
    batch_rainy, batch_targets = next(iter(train_loader))
    assert tuple(batch_rainy.shape) == (4, 3, 32, 32)
    assert tuple(batch_targets["clean"].shape) == (4, 3, 32, 32)
    assert tuple(batch_targets["rain_layer"].shape) == (4, 3, 32, 32)

    model = DerainingModel(ModelConfig(in_channels=3, hidden_channels=24, num_blocks=3))
    outputs = model(batch_rainy)
    assert set(outputs.keys()) == {"restored", "rain_layer"}
    assert tuple(outputs["restored"].shape) == (4, 3, 32, 32)
    assert tuple(outputs["rain_layer"].shape) == (4, 3, 32, 32)

    loss, parts = deraining_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"reconstruction_loss", "rain_loss"}
    assert float(parts["reconstruction_loss"]) >= 0.0
    assert float(parts["rain_loss"]) >= 0.0
    loss.backward()


def test_vision_image_deraining_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_60_synthetic_image_deraining.data import DataConfig
    from tracks.vision.lesson_60_synthetic_image_deraining.model import ModelConfig
    from tracks.vision.lesson_60_synthetic_image_deraining.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_deraining_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            image_size=32,
            val_fraction=0.2,
            seed=11,
            num_workers=0,
            in_channels=3,
            rain_lines_min=6,
            rain_lines_max=12,
            rain_strength_min=0.12,
            rain_strength_max=0.28,
        ),
        ModelConfig(in_channels=3, hidden_channels=24, num_blocks=3),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_60_synthetic_image_deraining" / "pytest_deraining_smoke"
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
        "train_reconstruction_loss",
        "train_rain_loss",
        "eval_loss",
        "eval_reconstruction_loss",
        "eval_rain_loss",
        "eval_psnr",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
