import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_image_dehazing_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_23_synthetic_image_dehazing.data import (
        DataConfig,
        SyntheticImageDehazingDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_23_synthetic_image_dehazing.model import (
        DehazingModel,
        ModelConfig,
        dehazing_loss,
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
    ds = SyntheticImageDehazingDataset(cfg)
    hazy, target = ds[0]

    assert tuple(hazy.shape) == (3, 32, 32)
    assert set(target.keys()) == {"clean", "transmission"}
    assert tuple(target["clean"].shape) == (3, 32, 32)
    assert tuple(target["transmission"].shape) == (1, 32, 32)
    assert hazy.dtype == torch.float32
    assert target["clean"].dtype == torch.float32
    assert target["transmission"].dtype == torch.float32
    assert 0.0 <= float(hazy.min().item()) <= float(hazy.max().item()) <= 1.0
    assert 0.0 <= float(target["transmission"].min().item()) <= float(target["transmission"].max().item()) <= 1.0

    train_loader, _ = get_dataloaders(cfg)
    batch_hazy, batch_targets = next(iter(train_loader))
    assert tuple(batch_hazy.shape) == (4, 3, 32, 32)
    assert tuple(batch_targets["clean"].shape) == (4, 3, 32, 32)
    assert tuple(batch_targets["transmission"].shape) == (4, 1, 32, 32)

    model = DehazingModel(ModelConfig(in_channels=3, hidden_channels=24, num_blocks=3))
    outputs = model(batch_hazy)
    assert set(outputs.keys()) == {"restored", "transmission"}
    assert tuple(outputs["restored"].shape) == (4, 3, 32, 32)
    assert tuple(outputs["transmission"].shape) == (4, 1, 32, 32)

    loss, parts = dehazing_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"reconstruction_loss", "transmission_loss"}
    assert float(parts["reconstruction_loss"]) >= 0.0
    assert float(parts["transmission_loss"]) >= 0.0
    loss.backward()


def test_vision_image_dehazing_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_23_synthetic_image_dehazing.data import DataConfig
    from tracks.vision.lesson_23_synthetic_image_dehazing.model import ModelConfig
    from tracks.vision.lesson_23_synthetic_image_dehazing.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))

    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_dehazing_smoke",
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
        ModelConfig(in_channels=3, hidden_channels=24, num_blocks=3),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_23_synthetic_image_dehazing" / "pytest_dehazing_smoke"
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
        "train_transmission_loss",
        "eval_loss",
        "eval_reconstruction_loss",
        "eval_transmission_loss",
        "eval_psnr",
    ):
        assert key in record
        assert float(record[key]) >= 0.0
