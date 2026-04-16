import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_vision_anomaly_detection_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_89_synthetic_anomaly_detection.data import (
        DataConfig,
        SyntheticVisionAnomalyDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_89_synthetic_anomaly_detection.model import (
        ModelConfig,
        anomaly_accuracy,
        anomaly_loss,
        build_model,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        image_size=32,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
        in_channels=3,
        anomaly_fraction=0.4,
        noise_std=0.01,
    )
    ds = SyntheticVisionAnomalyDataset(cfg)
    image, targets = ds[0]

    assert tuple(image.shape) == (3, 32, 32)
    assert set(targets.keys()) == {"reconstruction", "anomaly_map", "label"}
    assert tuple(targets["reconstruction"].shape) == (3, 32, 32)
    assert tuple(targets["anomaly_map"].shape) == (1, 32, 32)
    assert tuple(targets["label"].shape) == ()
    assert image.dtype == torch.float32
    assert targets["reconstruction"].dtype == torch.float32
    assert targets["anomaly_map"].dtype == torch.float32
    assert targets["label"].dtype == torch.float32

    train_loader, _ = get_dataloaders(cfg)
    batch_images, batch_targets = next(iter(train_loader))
    assert tuple(batch_images.shape) == (4, 3, 32, 32)
    assert tuple(batch_targets["reconstruction"].shape) == (4, 3, 32, 32)
    assert tuple(batch_targets["anomaly_map"].shape) == (4, 1, 32, 32)
    assert tuple(batch_targets["label"].shape) == (4,)

    model = build_model(
        ModelConfig(
            in_channels=3,
            arch="patchcore:patchcore_tiny",
            variant="",
            width_mult=1.0,
        )
    )
    outputs = model(batch_images)
    assert set(outputs.keys()) == {"reconstruction", "anomaly_map", "score"}
    assert tuple(outputs["reconstruction"].shape) == (4, 3, 32, 32)
    assert tuple(outputs["anomaly_map"].shape) == (4, 1, 32, 32)
    assert tuple(outputs["score"].shape) == (4,)

    loss, parts = anomaly_loss(outputs, batch_targets)
    assert torch.isfinite(loss)
    assert set(parts.keys()) == {"reconstruction_loss", "anomaly_map_l1", "score_bce"}
    assert float(parts["reconstruction_loss"]) >= 0.0
    assert float(parts["anomaly_map_l1"]) >= 0.0
    assert float(parts["score_bce"]) >= 0.0
    assert 0.0 <= anomaly_accuracy(outputs["score"], batch_targets["label"]) <= 1.0
    loss.backward()


def test_vision_anomaly_detection_training_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracks.vision.lesson_89_synthetic_anomaly_detection.data import DataConfig
    from tracks.vision.lesson_89_synthetic_anomaly_detection.model import ModelConfig
    from tracks.vision.lesson_89_synthetic_anomaly_detection.train import (
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
            run_name="pytest_vision_anomaly_smoke",
            arch="patchcore:patchcore_tiny",
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
            anomaly_fraction=0.4,
            noise_std=0.01,
        ),
        ModelConfig(
            in_channels=3,
            arch="patchcore:patchcore_tiny",
            variant="",
            width_mult=1.0,
        ),
    )

    assert exit_code == 0

    run_dir = tmp_path / "vision" / "lesson_89_synthetic_anomaly_detection" / "pytest_vision_anomaly_smoke"
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
    for key in ("train_loss", "train_anomaly_acc", "eval_loss", "eval_anomaly_acc"):
        assert key in record
        assert float(record[key]) >= 0.0
