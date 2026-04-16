import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_face_detection_batch_contract_and_loss_smoke() -> None:
    from tracks.vision.lesson_38_synthetic_face_detection.data import (
        DataConfig,
        SyntheticFaceDetectionDataset,
        get_dataloaders,
    )
    from tracks.vision.lesson_38_synthetic_face_detection.model import (
        FaceDetectionConfig,
        FaceDetectionModel,
        box_l1_error_pixels,
        detection_loss,
    )

    cfg = DataConfig(
        num_samples=32,
        batch_size=4,
        image_size=48,
        val_fraction=0.2,
        seed=0,
        num_workers=0,
    )

    dataset = SyntheticFaceDetectionDataset(cfg)
    image, box = dataset[0]
    assert tuple(image.shape) == (1, 48, 48)
    assert tuple(box.shape) == (4,)
    assert image.dtype == torch.float32
    assert box.dtype == torch.float32
    assert torch.all(box >= 0.0)
    assert torch.all(box <= 1.0)
    assert float(box[0]) <= float(box[2])
    assert float(box[1]) <= float(box[3])

    train_loader, _ = get_dataloaders(cfg)
    images, boxes = next(iter(train_loader))
    assert tuple(images.shape) == (4, 1, 48, 48)
    assert tuple(boxes.shape) == (4, 4)

    model = FaceDetectionModel(
        FaceDetectionConfig(in_channels=1, hidden_channels=24, num_blocks=3, dropout=0.0)
    )
    preds = model(images)
    assert tuple(preds.shape) == (4, 4)
    assert torch.all(preds >= 0.0)
    assert torch.all(preds <= 1.0)

    loss = detection_loss(preds, boxes)
    assert torch.isfinite(loss)
    assert box_l1_error_pixels(preds.detach(), boxes, image_size=48) >= 0.0
    loss.backward()


def test_face_detection_training_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tracks.vision.lesson_38_synthetic_face_detection.data import DataConfig
    from tracks.vision.lesson_38_synthetic_face_detection.model import FaceDetectionConfig
    from tracks.vision.lesson_38_synthetic_face_detection.train import TrainConfig, run_training

    monkeypatch.setenv("DLHUB_OUTPUTS_DIR", str(tmp_path))
    exit_code = run_training(
        TrainConfig(
            epochs=1,
            learning_rate=1e-3,
            seed=42,
            device="cpu",
            max_train_batches=2,
            max_eval_batches=1,
            run_name="pytest_face_detection_smoke",
        ),
        DataConfig(
            num_samples=48,
            batch_size=4,
            image_size=48,
            val_fraction=0.2,
            seed=5,
            num_workers=0,
        ),
        FaceDetectionConfig(in_channels=1, hidden_channels=24, num_blocks=3, dropout=0.0),
    )

    assert exit_code == 0
    run_dir = tmp_path / "vision" / "lesson_38_synthetic_face_detection" / "pytest_face_detection_smoke"
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
    assert record["epoch"] == 1
    assert float(record["train_loss"]) >= 0.0
    assert float(record["eval_loss"]) >= 0.0
    assert float(record["eval_l1_px"]) >= 0.0
